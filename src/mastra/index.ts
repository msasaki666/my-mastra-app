import { Mastra } from "@mastra/core";
import { google } from "@ai-sdk/google";
import { Agent } from "@mastra/core/agent";
import { PgVector } from "@mastra/pg";
import { createGraphRAGTool } from "@mastra/rag";
import {
	vectorStoreName,
	indexName,
	dimension,
	embeddingModelName,
} from "./settings";
import { LibSQLStore } from "@mastra/libsql";

type Environment = {
	google_generative_ai_api_key: string;
	postgres_connection_string: string;
};

const parseEnv = (): Environment => {
	const e = {
		google_generative_ai_api_key: process.env.GOOGLE_GENERATIVE_AI_API_KEY,
		postgres_connection_string: process.env.POSTGRES_CONNECTION_STRING,
	};
	if (!e.google_generative_ai_api_key || !e.postgres_connection_string) {
		throw new Error("Missing environment variables");
	}
	return e as Environment;
};

const env = parseEnv();
const embeddingModel = google.textEmbeddingModel(embeddingModelName, {
	taskType: "RETRIEVAL_QUERY", // default value
});
const graphRagTool = createGraphRAGTool({
	vectorStoreName,
	indexName,
	model: embeddingModel,
	graphOptions: {
		dimension,
		threshold: 0.7,
	},
});

const ragAgent = new Agent({
	name: "GraphRAG Agent",
	instructions: `あなたは、提供されたコンテキストに基づいて質問に答える役立つアシスタントです。回答を次のようにフォーマットしてください：

1. 直接的な事実: 質問に関連するテキストから直接述べられている事実のみをリストアップします（2-3の箇条書き）
2. 作られたつながり: テキストの異なる部分間で見つけた関係をリストアップします（2-3の箇条書き）
3. 結論: すべてをまとめる1文の要約

各セクションを簡潔にし、最も重要なポイントに焦点を当ててください。

重要: 質問に答えるよう求められた場合、ツールで提供されたコンテキストのみに基づいて回答してください。
コンテキストに質問に完全に答えるための十分な情報が含まれていない場合は、その旨を明示してください。`,
	model: google.chat("gemini-2.0-flash"),
	tools: {
		graphRagTool,
	},
});

const pgVector = new PgVector({
	connectionString: env.postgres_connection_string,
});

export const mastra = new Mastra({
	agents: { ragAgent },
	vectors: { pgVector },
	server: {
		host: "0.0.0.0",
	},
	storage: new LibSQLStore({
		url: "file:../mastra.db",
	}),
});
