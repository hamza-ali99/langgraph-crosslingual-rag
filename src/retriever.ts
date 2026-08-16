import { tool } from "@langchain/core/tools";
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
import { Document } from "@langchain/core/documents";
import * as z from "zod";
import { CohereRerank } from "@langchain/cohere";
import { ChatGroq } from "@langchain/groq";
import { BaseMessage } from "@langchain/core/messages";

const translator = new ChatGroq({
  model: "llama-3.1-8b-instant",
  apiKey: process.env.GROQ_API_KEY!,
  temperature: 0,
});

const TRANSLATION_PROMPT = `\
Translate the following English car-related question into Japanese.
Output ONLY the Japanese translation, nothing else.

English: {query}
Japanese: `;

const REWRITE_PROMPT = `\
Given this conversation history and a follow-up question, rewrite the follow-up into a standalone question that can be understood without any prior context. The standalone question should be specific and include all necessary details from the conversation.

Conversation history:
{history}

Follow-up question: {query}

Standalone question:`;

async function translateToJapanese(query: string): Promise<string> {
  try {
    const prompt = TRANSLATION_PROMPT.replace("{query}", query);
    const res = await translator.invoke([{ role: "user", content: prompt }]);
    return (res.content as string).trim();
  } catch (err) {
    console.warn("Translation failed, using original query:", err);
    return query;
  }
}

export async function rewriteQuery(query: string, history: BaseMessage[]): Promise<string> {
  if (history.length === 0) return query;

  const historyWithoutCurrent = history.slice(0, -1);
  if (historyWithoutCurrent.length === 0) return query;

  const historyText = historyWithoutCurrent
    .map((m) => {
      const role = m._getType?.() === "human" ? "User" : "Assistant";
      return `${role}: ${m.content}`;
    })
    .join("\n");

  try {
    const prompt = REWRITE_PROMPT
      .replace("{history}", historyText)
      .replace("{query}", query);
    const res = await translator.invoke([{ role: "user", content: prompt }]);
    const rewritten = (res.content as string).trim();
    console.log(`[REWRITER] "${query}" → "${rewritten}"`);
    return rewritten;
  } catch (err) {
    console.warn("Query rewriting failed, using original:", err);
    return query;
  }
}

function sanitizeQuery(query: string): string {
  return query
    .replace(/[\x00-\x1F\x7F]/g, "")
    .replace(/\s+/g, " ")
    .trim();
}

export function createRetrieverTool(vectorStore: MemoryVectorStore) {
  return tool(
    async ({ query }) => {
      try {
        const sanitized = sanitizeQuery(query);

        if (!sanitized) {
          return "Invalid query. Please ask a question about your Nissan Moco 2012.";
        }

        const japanese = await translateToJapanese(sanitized);
        const expanded = sanitized + " " + japanese;
        console.log(`[RETRIEVER] Expanded: "${expanded}"`);

        const baseRetriever = vectorStore.asRetriever({ k: 100 });
        const initialDocs = await baseRetriever.invoke(expanded);

        if (!initialDocs.length) {
          return "No relevant section found in the Moco manual for this query.";
        }

        const reranker = new CohereRerank({
          apiKey: process.env.COHERE_API_KEY!,
          model: "rerank-multilingual-v3.0",
          topN: 8,
        });

        const rerankedDocs = await reranker.compressDocuments(initialDocs, sanitized);

        console.log(`[RETRIEVER] Got ${initialDocs.length} initial docs, ${rerankedDocs.length} after rerank`);

        return rerankedDocs
          .map((doc: Document) =>
            `[Page ${doc.metadata.pageNumber}]\n${doc.pageContent.slice(0, 800)}`
          )
          .join("\n\n---\n\n");
      } catch (err) {
        console.error("❌ Retriever error:", err);
        return "Retrieval failed.";
      }
    },
    {
      name: "retrieve_from_manual",
      description:
        "Search the Nissan Moco 2012 Japanese owner's manual for relevant sections. Use this for any question about the car.",
      schema: z.object({
        query: z.string().describe("The question to search for in the manual"),
      }),
    }
  );
}
