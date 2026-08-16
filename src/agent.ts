import "dotenv/config";
import { StateGraph, Annotation, MessagesAnnotation } from "@langchain/langgraph";
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
import { createRetrieverTool, rewriteQuery } from "./retriever.ts";
import { ChatGroq } from "@langchain/groq";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";

const SYSTEM_PROMPT = `You are an assistant for a Nissan Moco 2012 owner who cannot read Japanese.

You will be provided with sections retrieved from the Japanese owner's manual. Base your answer EXCLUSIVELY on these retrieved sections.

**CRITICAL RULES — FOLLOW EXACTLY:**

1. Answer ENTIRELY in English — translate ALL Japanese text including product names.
2. ALWAYS cite page numbers exactly as they appear in the retrieved chunks. The chunks show page numbers in brackets like [Page 13]. Use that exact number. Do NOT invent page numbers.
3. Only cite pages that are present in the retrieved chunks. If you are not sure, do not cite any page.
4. Do NOT add any actions, explanations, warnings, or steps that are not explicitly written in the retrieved text.
5. If the manual only lists causes or conditions, simply repeat what the manual says. Do not infer missing steps.
6. Do NOT combine information from different sections unless both are present in the retrieved chunks.
7. **CRITICAL: If the retrieved text contains a bullet point saying "see Maintenance Note" or similar, IGNORE that reference.** The actual specifications are often in TABLES on the SAME PAGE. Check the FULL retrieved text including all tables before concluding the answer is missing.
8. **CRITICAL: Respect section headings.** The manual is organised into sections with specific headings. Information under one heading applies ONLY to that section. Do NOT transfer specifications (ratios, capacities, temperatures, or procedures) from one section to another, even if they seem related.
9. Never invent specifications, numbers, or procedures. Only use what is verbatim in the retrieved text.
10. **NEVER add disclaimers, caveats, or "consult the manual / Nissan representative" notes at the end of your answer.** If the answer is in the retrieved text, state it directly and stop. If not, use ONLY the exact fallback sentence provided below.
11. **NEVER include Japanese text, kanji, hiragana, or katakana in your answer.** Do not quote Japanese phrases, do not put Japanese in parentheses, and do not provide transliterations or romanizations. Translate everything to English and output English only.
12. **Each answer must be based ONLY on the retrieved sections for THIS question.** Do not reference or build upon previous answers. If the retrieved sections do not mention a topic, say "I couldn't find this" even if a previous answer discussed it.

**If the retrieved sections do not contain the answer after checking all tables and text, say exactly:**
"I couldn't find this in your Moco manual. You may want to contact a Nissan dealer or check the relevant section in the manual."

Now follow these rules strictly. Do not add any extra information.`;

const GraphState = Annotation.Root({
  ...MessagesAnnotation.spec,
  query: Annotation<string>({
    reducer: (_prev, next) => next,
    default: () => "",
  }),
  retrievedContext: Annotation<string>({
    reducer: (_prev, next) => next,
    default: () => "",
  }),
});

export async function createMocoAgent(vectorStore: MemoryVectorStore) {
  const llm = new ChatGroq({
    model: "llama-3.3-70b-versatile",
    apiKey: process.env.GROQ_API_KEY!,
    temperature: 0,
  });

  const retrieverTool = createRetrieverTool(vectorStore);

  async function rewriteNode(state: typeof GraphState.State) {
    const messages = state.messages;
    const lastMsg = messages[messages.length - 1];
    let standaloneQuery = lastMsg.content as string;

    if (messages.length > 1 && lastMsg._getType?.() === "human") {
      standaloneQuery = await rewriteQuery(lastMsg.content as string, messages);
    }
    return { query: standaloneQuery };
  }

  async function retrieveNode(state: typeof GraphState.State) {
    const docs = await retrieverTool.invoke({ query: state.query });
    return { retrievedContext: docs };
  }

  async function generateNode(state: typeof GraphState.State) {
    // Stateless generation: only the standalone query + retrieved docs
    // No conversation history to prevent cross-turn contamination
    const prompt = [
      new SystemMessage(SYSTEM_PROMPT),
      new HumanMessage(state.query),
      new HumanMessage(
        `Retrieved manual sections:\n\n${state.retrievedContext}\n\nAnswer the question based ONLY on the retrieved sections above.`
      ),
    ];
    const response = await llm.invoke(prompt);
    return { messages: [response] };
  }

  const graph = new StateGraph(GraphState)
    .addNode("rewrite", rewriteNode)
    .addNode("retrieve", retrieveNode)
    .addNode("generate", generateNode)
    .addEdge("__start__", "rewrite")
    .addEdge("rewrite", "retrieve")
    .addEdge("retrieve", "generate")
    .addEdge("generate", "__end__")
    .compile();

  return graph;
}
