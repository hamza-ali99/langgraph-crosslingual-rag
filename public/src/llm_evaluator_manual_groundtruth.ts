import "dotenv/config";
import { buildVectorStore } from "./ingest";
import { createMocoAgent } from "./agent";
import { createRetrieverTool } from "./retriever.ts";
import { HumanMessage } from "@langchain/core/messages";
import { ChatGroq } from "@langchain/groq";
import * as fs from "fs";

interface TestCase {
  id: string;
  question: string;
  category: string;
}

const TEST_SET: TestCase[] = [
  { id: "TC001", question: "What is the fuel tank capacity?", category: "specs" },
  { id: "TC002", question: "What type of coolant does the car use?", category: "fluids" },
  { id: "TC003", question: "What should I do if the engine overheats?", category: "emergency" },
  { id: "TC004", question: "What is the recommended tire pressure?", category: "specs" },
  { id: "TC005", question: "How often should I change the oil?", category: "maintenance" },
  { id: "TC006", question: "What should I do when the warning buzzer sounds?", category: "warning" },
  { id: "TC007", question: "How do I open and close the doors?", category: "operation" },
  { id: "TC008", question: "How do I adjust the seats?", category: "seats" },
  { id: "TC009", question: "What is the SRS airbag system and when does it deploy?", category: "safety" },
  { id: "TC010", question: "How do I use the windshield wipers?", category: "controls" },
  { id: "TC011", question: "What is ABS and how does it work?", category: "braking" },
  { id: "TC012", question: "How do I shift gears in an automatic transmission?", category: "driving" },
  { id: "TC013", question: "What is the torque specification for the lug nuts?", category: "fallback" },
  { id: "TC014", question: "Is the Nissan Moco 2012 electric?", category: "out-of-scope" },
  { id: "TC015", question: "What does the S or Sports mode do?", category: "driving" },
  { id: "TC016", question: "How does engine braking work?", category: "driving" },
  { id: "TC017", question: "What is the L gear used for?", category: "driving" },
];

const EVALUATION_PROMPT = `
You are a strict QA evaluator. The assistant's answer must be judged only against the manual content provided below.

Question: {question}
Assistant Answer: {answer}
Manual Content (the exact text the assistant had access to): {manualContent}

Evaluate on three metrics (0-10):

1. Faithfulness: Is every claim in the assistant's answer directly supported by the manual content? (10 = fully supported, 0 = any unsupported claim)

2. Correctness: Is the answer factually accurate according to the manual? (10 = perfect, 0 = factually wrong)

3. Citation accuracy: For each page number the assistant cites (e.g., "page 137", "(page 2-28)", "[Page 13]"), check if that page number appears anywhere in the Manual Content. If ANY cited page is not present in the Manual Content, citation accuracy = 0. If all cited pages appear, score 10. If no pages are cited, score 10.

Important:
- Page numbers may appear in two formats: raw PDF page (e.g., "[Page 42]") or printed chapter-page (e.g., "2-28"). Treat them as referring to the same content if the manual content contains the raw page number that corresponds to that printed page. You can infer the mapping if the chunk includes both.
- If the Manual Content does NOT contain the answer, the assistant should say "I couldn't find this". Then faithfulness=10, correctness=10, citation=10.
- If the assistant says "I couldn't find this" but the Manual Content DOES contain the answer, faithfulness=0.

Output ONLY valid JSON. Do not wrap in markdown fences. Example: {"faithfulness":10,"correctness":10,"citation_accuracy":10}
`;

function deterministicCitationCheck(answer: string, manualContent: string): number {
  const pageMatches = answer.matchAll(/page\s+([\d\-]+)/gi);
  const citedPages = new Set<string>();
  for (const match of pageMatches) {
    citedPages.add(match[1].toLowerCase());
  }
  if (citedPages.size === 0) return 10;

  let valid = 0;
  for (const page of citedPages) {
    const rawPattern = new RegExp(`\\[Page\\s+${page}\\]`, 'i');
    const standalonePattern = new RegExp(`\\b${page.replace('-', '\\-')}\\b`, 'i');
    if (rawPattern.test(manualContent) || standalonePattern.test(manualContent)) {
      valid++;
    }
  }
  return Math.round((valid / citedPages.size) * 10);
}

async function evaluate() {
  console.log("Loading RAG system...");
  const vectorStore = await buildVectorStore();
  const agent = await createMocoAgent(vectorStore);
  const retrieverTool = createRetrieverTool(vectorStore);

  const judge = new ChatGroq({
    model: "llama-3.3-70b-versatile",
    apiKey: process.env.GROQ_API_KEY!,
    temperature: 0,
  });

  const runs = process.argv.includes("--runs")
    ? parseInt(process.argv[process.argv.indexOf("--runs") + 1])
    : 3;

  const onlyTestId = process.argv.find(arg => arg.startsWith("--test="))?.split("=")[1];
  let testSet = onlyTestId ? TEST_SET.filter(tc => tc.id === onlyTestId) : TEST_SET;

  if (!testSet.length) {
    console.error("No test cases found for filter:", onlyTestId);
    return;
  }

  const allResults: any[] = [];

  for (let run = 1; run <= runs; run++) {
    console.log(`\n${"=".repeat(60)}`);
    console.log(`RUN ${run} / ${runs}`);
    console.log(`${"=".repeat(60)}`);

    for (const tc of testSet) {
      console.log(`\n🧪 Testing ${tc.id}: "${tc.question}" (${tc.category})`);

      // 1. Run the agent (StateGraph returns retrievedContext in state)
      const agentResult = await agent.invoke({ messages: [new HumanMessage(tc.question)] });
      const answer = agentResult.messages.at(-1)?.content as string;

      // 2. Get the EXACT context the agent saw from graph state
      let manualContent = (agentResult as any).retrievedContext || "";
      if (!manualContent.trim()) {
        console.warn(`⚠️ No retrieved context for ${tc.id}, falling back to re-retrieval`);
        manualContent = await retrieverTool.invoke({ query: tc.question });
      }

      console.log(`\n📝 ANSWER:\n${answer}\n`);
      console.log(`📚 MANUAL (first 600 chars):\n${manualContent.substring(0, 600)}...\n`);

      // 3. Deterministic citation check
      const detCitationScore = deterministicCitationCheck(answer, manualContent);
      console.log(`🔍 Deterministic citation check: ${detCitationScore}/10`);

      // 4. LLM judge evaluation
      const prompt = EVALUATION_PROMPT
        .replace("{question}", tc.question)
        .replace("{answer}", answer)
        .replace("{manualContent}", manualContent);

      let judgeRes;
      try {
        judgeRes = await judge.invoke([{ role: "user", content: prompt }]);
      } catch (err) {
        console.error(`Judge failed for ${tc.id}:`, err);
        continue;
      }

      let raw = judgeRes.content as string;
      raw = raw.replace(/```json\s*|```\s*/g, "").trim();

      let scores;
      try {
        scores = JSON.parse(raw);
      } catch (e) {
        console.error(`JSON parse error for ${tc.id}:`, raw);
        scores = { faithfulness: 0, correctness: 0, citation_accuracy: 0 };
      }

      const finalCitation = Math.min(scores.citation_accuracy, detCitationScore);

      console.log(`🤖 Judge: Faith=${scores.faithfulness}, Correct=${scores.correctness}, Citation=${scores.citation_accuracy} (final: ${finalCitation})`);

      allResults.push({
        run,
        id: tc.id,
        question: tc.question,
        faithfulness: scores.faithfulness,
        correctness: scores.correctness,
        citation_llm: scores.citation_accuracy,
        citation_deterministic: detCitationScore,
        citation_final: finalCitation,
      });

      const logContent = `
${"=".repeat(80)}
${tc.id} | ${tc.question} | Run ${run}
${"=".repeat(80)}
ASSISTANT ANSWER:
${answer}

MANUAL CONTENT (full):
${manualContent}

JUDGE RAW OUTPUT:
${raw}

DETERMINISTIC CITATION SCORE: ${detCitationScore}
`;
      fs.writeFileSync(`debug_${tc.id}_run${run}.log`, logContent);
    }
  }

  const summary: any = {};
  for (const res of allResults) {
    if (!summary[res.id]) {
      summary[res.id] = {
        faith: [],
        correct: [],
        citation_llm: [],
        citation_det: [],
        citation_final: [],
      };
    }
    summary[res.id].faith.push(res.faithfulness);
    summary[res.id].correct.push(res.correctness);
    summary[res.id].citation_llm.push(res.citation_llm);
    summary[res.id].citation_det.push(res.citation_deterministic);
    summary[res.id].citation_final.push(res.citation_final);
  }

  console.log("\n" + "=".repeat(80));
  console.log("📊 FINAL REPORT (averaged over runs)");
  console.log("=".repeat(80));

  let totalFaith = 0, totalCorrect = 0, totalCitationFinal = 0, count = 0;
  for (const [id, m] of Object.entries(summary)) {
    const avgFaith = (m as any).faith.reduce((a: number, b: number) => a + b, 0) / (m as any).faith.length;
    const avgCorrect = (m as any).correct.reduce((a: number, b: number) => a + b, 0) / (m as any).correct.length;
    const avgCitationFinal = (m as any).citation_final.reduce((a: number, b: number) => a + b, 0) / (m as any).citation_final.length;
    console.log(`\n${id} (${TEST_SET.find(t => t.id === id)?.category})`);
    console.log(`  Faithfulness: ${avgFaith.toFixed(1)}/10`);
    console.log(`  Correctness:  ${avgCorrect.toFixed(1)}/10`);
    console.log(`  Citation (final): ${avgCitationFinal.toFixed(1)}/10`);
    totalFaith += avgFaith; totalCorrect += avgCorrect; totalCitationFinal += avgCitationFinal; count++;
  }

  console.log("\n" + "=".repeat(80));
  console.log("🏆 OVERALL AVERAGES");
  console.log("=".repeat(80));
  console.log(`Faithfulness: ${(totalFaith / count).toFixed(1)}/10`);
  console.log(`Correctness:  ${(totalCorrect / count).toFixed(1)}/10`);
  console.log(`Citation (final): ${(totalCitationFinal / count).toFixed(1)}/10`);

  const pass = (totalFaith / count) >= 7 && (totalCorrect / count) >= 7;
  console.log(`\n✅ Overall PASS? ${pass ? "YES" : "NO"} (threshold: faithfulness>=7 & correctness>=7)`);

  console.log("\n📁 Individual debug logs saved as debug_<TCID>_run<run>.log");
}

evaluate().catch(console.error);