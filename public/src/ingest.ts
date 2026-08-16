import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
import { CohereEmbeddings } from "@langchain/cohere";

import * as dotenv from "dotenv";
dotenv.config();

export async function buildVectorStore() {
  const loader = new PDFLoader("./data/moco-2012.pdf");
  const rawDocs = await loader.load();

  const numberedDocs = rawDocs.map((doc, index) => ({
    ...doc,
    metadata: { ...doc.metadata, page: index }
  }));

  const docs = numberedDocs
    .filter(doc => doc.pageContent.trim().length > 100)
    .filter(doc => doc.metadata.page > 3)
    .filter(doc => {
      const dotLeaderCount = (doc.pageContent.match(/･･/g) || []).length;
      return dotLeaderCount < 3;
    });

  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 2500,
    chunkOverlap: 500,
    separators: ["\n\n", "\n", "。", "、", "！", "？", ""],
  });
  const chunks = await splitter.splitDocuments(docs);
  console.log(`Created ${chunks.length} chunks`);

  const enrichedChunks = chunks.map(chunk => ({
    ...chunk,
    metadata: {
      ...chunk.metadata,
      pageNumber: (chunk.metadata.page ?? 0) + 1,
      source: "Nissan Moco 2012 Owner's Manual",
    }
  }));

  const embeddings = new CohereEmbeddings({
    model: "embed-multilingual-v3.0",
    apiKey: process.env.COHERE_API_KEY!,
  });

  const vectorStore = await MemoryVectorStore.fromDocuments(enrichedChunks, embeddings);
  console.log("Ingestion complete ✓");

  return vectorStore;
}