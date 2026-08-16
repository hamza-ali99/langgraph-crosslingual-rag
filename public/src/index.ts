import "dotenv/config";
import * as readline from "readline";
import { buildVectorStore } from "./ingest";
import { createMocoAgent } from "./agent";
import { HumanMessage, BaseMessage } from "@langchain/core/messages";

async function main() {
  console.log("Loading and indexing Moco manual, please wait...");
  const vectorStore = await buildVectorStore();
  const agent = await createMocoAgent(vectorStore);
  console.log("Ready! Ask anything about your Nissan Moco 2012.\n");

  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  const sessionMessages: BaseMessage[] = [];

  const ask = () => {
    rl.question("You: ", async (input) => {
      const question = input.trim();

      if (!question) { ask(); return; }
      if (question.toLowerCase() === "exit") {
        console.log("Goodbye!");
        rl.close();
        return;
      }

      try {
        sessionMessages.push(new HumanMessage(question));

        // Pass FULL history — the graph handles trimming internally
        const result = await agent.invoke({
          messages: sessionMessages,
        });

        const lastMessage = result.messages.at(-1);
        const answer = lastMessage?.content as string;
        console.log(`\nAssistant: ${answer}\n`);

        if (lastMessage) {
          sessionMessages.push(lastMessage);
        }

        while (sessionMessages.length > 6) {
          sessionMessages.shift();
        }
      } catch (err) {
        console.error("Error:", err);
      }

      ask();
    });
  };

  ask();
}

main().catch(console.error);