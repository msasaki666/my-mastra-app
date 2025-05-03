import { google } from "@ai-sdk/google";
import { embedMany } from "ai";
import {
	dimension,
	embeddingModelName,
	indexName,
	vectorStoreName,
} from "./settings";
import type { MDocument } from "@mastra/rag";

// チャンクの埋め込みを生成し、それをベクターデータベースに保存します:
const { embeddings } = await embedMany({
	model: google.textEmbeddingModel(embeddingModelName),
	values: chunks.map((chunk) => chunk.text),
});

const vectorStore = mastra.getVector(vectorStoreName);
await vectorStore.createIndex({
	indexName,
	dimension,
});
await vectorStore.upsert({
	indexName,
	vectors: embeddings,
	metadata: chunks?.map((chunk) => ({ text: chunk.text })),
});
