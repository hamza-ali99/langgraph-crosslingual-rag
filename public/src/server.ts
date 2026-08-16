import "dotenv/config";
import express from "express";
import cors from "cors";
import { buildVectorStore } from "./ingest";
import { createMocoAgent } from "./agent";
import { HumanMessage, BaseMessage } from "@langchain/core/messages";

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());
app.use(express.static("public"));

let agent: any = null;

const sessions = new Map<string, BaseMessage[]>();

function getSession(sessionId: string): BaseMessage[] {
  if (!sessions.has(sessionId)) {
    sessions.set(sessionId, []);
  }
  return sessions.get(sessionId)!;
}

function trimSession(messages: BaseMessage[]): BaseMessage[] {
  const MAX_MESSAGES = 6;
  if (messages.length > MAX_MESSAGES) {
    return messages.slice(-MAX_MESSAGES);
  }
  return messages;
}

async function initAgent() {
  console.log("🔄 Loading and indexing manual...");
  const vectorStore = await buildVectorStore();
  agent = await createMocoAgent(vectorStore);
  console.log("✅ Agent ready! Visit http://localhost:" + PORT);
}

app.post("/ask", async (req, res) => {
  if (!agent) {
    return res.status(503).json({ error: "Agent not ready yet" });
  }

  const { question, sessionId } = req.body;
  if (!question || typeof question !== "string") {
    return res.status(400).json({ error: "Missing question" });
  }

  const sid = sessionId || crypto.randomUUID();
  const sessionMessages = getSession(sid);

  try {
    sessionMessages.push(new HumanMessage(question));

    const result = await agent.invoke({
      messages: sessionMessages,
    });

    const lastMessage = result.messages.at(-1);

    if (lastMessage) {
      sessionMessages.push(lastMessage);
    }

    const trimmed = trimSession(sessionMessages);
    sessions.set(sid, trimmed);

    res.json({
      answer: lastMessage?.content || "No response",
      sessionId: sid,
    });
  } catch (err) {
    console.error("Error:", err);
    res.status(500).json({ error: "Internal server error" });
  }
});

app.listen(PORT, () => {
  console.log(`🚀 UI server running at http://localhost:${PORT}`);
  initAgent();
});