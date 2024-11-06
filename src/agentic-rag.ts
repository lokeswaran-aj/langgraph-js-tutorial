import { CheerioWebBaseLoader } from '@langchain/community/document_loaders/web/cheerio';
import { AIMessage, BaseMessage, HumanMessage } from '@langchain/core/messages';
import { ChatPromptTemplate } from '@langchain/core/prompts';
import { Annotation, END, START, StateGraph } from '@langchain/langgraph';
import { ToolNode } from '@langchain/langgraph/prebuilt';
import { ChatOpenAI, OpenAIEmbeddings } from '@langchain/openai';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import { config } from 'dotenv';
import { pull } from "langchain/hub";
import { createRetrieverTool } from 'langchain/tools/retriever';
import { MemoryVectorStore } from 'langchain/vectorstores/memory';
import { z } from 'zod';

config()

const main = async () => {
  const urls = [
    'https://lilianweng.github.io/posts/2023-06-23-agent/',
    'https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/',
    'https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/',
  ]

  const docs = await Promise.all(
    urls.map((url) => new CheerioWebBaseLoader(url).load()),
  )
  const docLists = docs.flat()

  const textsplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 500,
    chunkOverlap: 50,
  })

  const docSplits = await textsplitter.splitDocuments(docLists)

  const vectorstore = new MemoryVectorStore(
    new OpenAIEmbeddings({ model: 'text-embedding-3-small' }),
  )

  await vectorstore.addDocuments(docSplits)

  const retriever = vectorstore.asRetriever()

  const MessageState = Annotation.Root({
    messages: Annotation<BaseMessage[]>({
      reducer: (x, y) => x.concat(y),
      default: () => [],
    }),
  })

  type MessageStateType = typeof MessageState.State

  const retrieverTool = createRetrieverTool(retriever, {
    name: 'retrieve_blog_posts',
    description:
      'Search and return information about Lilian Weng blog posts on LLM agents, prompt engineering, and adversarial attacks on LLMs.',
  })
  const tools = [retrieverTool]

  const toolNode = new ToolNode<MessageStateType>(tools)

  // edges
  const shouldRetrieve = (state: MessageStateType): string => {
    const { messages } = state
    console.log('---DECIDE TO RETRIEVE')
    const lastMessage = messages[messages.length - 1]

    if (
      'tool_calls' in lastMessage &&
      Array.isArray(lastMessage.tool_calls) &&
      lastMessage.tool_calls.length
    ) {
      console.log('---DECISION: RETRIEVE')
      return 'retrieve'
    }

    console.log('---DECISION: END')
    return END
  }

  const gradeDocuments = async (
    state: MessageStateType,
  ): Promise<Partial<MessageStateType>> => {
    const { messages } = state

    const tool = {
      name: 'give_relevance_score',
      description: 'Give a relevance score to the retrieved documents.',
      schema: z.object({
        binaryScore: z.string().describe("Relevance score 'yes' or 'no'"),
      }),
    }

    const prompt = ChatPromptTemplate.fromTemplate(
      `You are a grader assessing relevance of retrieved docs to a user question.
  Here are the retrieved docs:
  \n ------- \n
  {context} 
  \n ------- \n
  Here is the user question: {question}
  If the content of the docs are relevant to the users question, score them as relevant.
  Give a binary score 'yes' or 'no' score to indicate whether the docs are relevant to the question.
  Yes: The docs are relevant to the question.
  No: The docs are not relevant to the question.`,
    )

    const llm = new ChatOpenAI({
      model: 'gpt-4o-mini',
      temperature: 0,
    }).bindTools([tool], { tool_choice: tool.name })

    const chain = prompt.pipe(llm)

    const lastMessage = messages[messages.length - 1]

    const score = await chain.invoke({
      question: messages[0].content as string,
      context: lastMessage.content as string,
    })

    console.log(`---SCORE:${score.content}`)

    return {
      messages: [score],
    }
  }

  const checkRelevance = (state: MessageStateType): string => {
    const { messages } = state
    console.log('---CHECK RELEVANCE---')
    const lastMessage = messages[messages.length - 1]
    if (!('tool_calls' in lastMessage)) {
      throw new Error(
        "The 'checkRelevance' node requires the most recent message to contain tool calls.",
      )
    }
    const toolCalls = (lastMessage as AIMessage).tool_calls
    if (!toolCalls || !toolCalls.length)
      throw new Error('Last message was not a function message')

    if (toolCalls[0].args.binaryScore === 'yes') {
      console.log('---DECISIION: YES, DOCS REVELANT---')
      return 'yes'
    }
    console.log('---DECISIION: NO, DOCS NOT REVELANT---')
    return 'no'
  }

  // nodes
  const agent = async (state: MessageStateType) => {
    const { messages } = state
    console.log('---CALLING AGENT---')
    const filteredMessages = messages.filter((message) => {
      if (
        'tool_calls' in message &&
        Array.isArray(message.tool_calls) &&
        message.tool_calls.length > 0
      )
        return message.tool_calls[0].name !== 'give_relevance_score'
      return true
    })
    const llm = new ChatOpenAI({
      model: 'gpt-4o-mini',
      temperature: 0,
    }).bindTools(tools)

    const response = await llm.invoke(filteredMessages)
    return {
      messages: [response],
    }
  }

  const rewrite = async (state: MessageStateType) => {
    const { messages } = state
    console.log('---TRANSFORMING QUERY---')
    const question = messages[0].content as string
    const prompt =
      ChatPromptTemplate.fromTemplate(`Look at the input and try to reason about the underlying semantic intent / meaning. \n 
Here is the initial question:
\n ------- \n
{question} 
\n ------- \n
Formulate an improved question:`)

    const llm = new ChatOpenAI({
      model: 'gpt-4o-mini',
      temperature: 0,
      streaming: true,
    })

    const response = await prompt.pipe(llm).invoke({ question })
    return {
      messages: [response],
    }
  }

  const generate = async (state: MessageStateType) => {
    const { messages } = state
    console.log('---GENERATING---')
    const question = messages[0].content as string
    const lastToolMessage = messages
      .slice()
      .reverse()
      .find((msg) => msg.getType() === 'tool')
    if (!lastToolMessage)
      throw new Error(
        "No tool message found in the conversation history.",
      )
    const docs = lastToolMessage.content as string
    const prompt = await pull('rlm/rag-prompt')
    const llm = new ChatOpenAI({
      model: 'gpt-4o-mini',
      temperature: 0,
      streaming: true,
    })

    const response = await prompt.pipe(llm).invoke({ context:docs, question })
    return {
      messages: [response],
    }
  }

  const workflow = new StateGraph(MessageState)
    .addNode('agent', agent)
    .addNode('retrieve', toolNode)
    .addNode('gradeDocuments', gradeDocuments)
    .addNode("rewrite", rewrite)
    .addNode("generate", generate)

  workflow.addEdge(START, 'agent')
  workflow.addConditionalEdges("agent", shouldRetrieve)
  workflow.addEdge("retrieve", "gradeDocuments")
  workflow.addConditionalEdges("gradeDocuments", checkRelevance, {
    yes: "generate",
    no: "rewrite",
  })
  workflow.addEdge('generate', END)
  workflow.addEdge('rewrite', "agent")

  const app = workflow.compile()

  const inputs = {
    messages: [new HumanMessage("What are the types of agent memory based on Lilian Weng's blog post?")]
  }

  let finalState;

  for await (const output of await app.stream(inputs)) { 
    for (const [key, value] of Object.entries(output)) { 
      const lastMsg = output[key].messages[output[key].messages.length - 1]
      console.log(`Output from node: '${key}`)
      console.dir({
        type: lastMsg.getType(),
        content: lastMsg.content,
        tool_calls: lastMsg.tool_calls,
      }, {depth: null})
      console.log("---\n")
      finalState = value
    }
  }
  console.log(JSON.stringify(finalState, null, 2))
}
main()
