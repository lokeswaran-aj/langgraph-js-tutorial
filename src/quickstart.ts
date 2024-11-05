// Optional, add tracing in LangSmith
import { TavilySearchResults } from '@langchain/community/tools/tavily_search'
import { MemorySaver } from '@langchain/langgraph'
import { createReactAgent } from '@langchain/langgraph/prebuilt'
import { ChatOpenAI } from '@langchain/openai'
import { config } from 'dotenv'

config()

const main = async () => {
  const tavilyTool = new TavilySearchResults({ maxResults: 2 })

  const llm = new ChatOpenAI({
    modelName: 'gpt-4o-mini',
    temperature: 0.0,
  })

  const checkpointSaver = new MemorySaver()

  const agent = createReactAgent({ llm, tools: [tavilyTool], checkpointSaver })

   await agent.invoke(
    { messages: [{ role: 'user', content: 'What is the weather in Pondicherry?' }] },
    { configurable: { thread_id: '1' } },
  )

  const agentFinalState = await agent.invoke(
    { messages: [{ role: 'user', content: 'What is the weather in SF?' }] },
    { configurable: { thread_id: '1' } },
  )

  agentFinalState.messages.map((message: any) => {
    console.warn("🚀 --------------------------------------🚀")
    console.warn("🚀 - message:", message, ": ", message.content)
    console.warn("🚀 --------------------------------------🚀")
  })
}

main()
