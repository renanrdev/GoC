const express = require('express');
const multer = require('multer');
const cors = require('cors');
const { Anthropic } = require('@anthropic-ai/sdk');
const { OpenAI } = require('openai');
const { GoogleGenerativeAI } = require('@google/generative-ai');
const fs = require('fs');
const path = require('path');
const dotenv = require('dotenv');
const swaggerUi = require('swagger-ui-express');
const swaggerSpec = require('./swagger');

const { 
  extractTextFromImage, 
  extractQuestion,
  formatAnalysisResult 
} = require('./imageProcessor');
const ocrConfig = require('./ocrConfig');

// Carregar variáveis de ambiente
dotenv.config();

//------------------------------------------------------------------------
// CONFIGURAÇÃO GLOBAL DOS MODELOS DE IA
// Altere estas variáveis para mudar os modelos utilizados
//------------------------------------------------------------------------

// Configuração dos modelos Claude (em ordem de preferência)
const CLAUDE_MODELS = [
  "claude-3-7-sonnet-20250219",  // Claude mais recente e avançado
  "claude-3-5-sonnet-20241022",  // Claude 3.5 Sonnet (nova versão)
  "claude-3-5-haiku-20241022",   // Claude 3.5 Haiku (mais rápido)
];

// Configuração dos modelos GPT (em ordem de preferência)
const GPT_MODELS = [
  "gpt-4.5-preview",
  "gpt-4o",
  "gpt-3.5-turbo",

];

const XAI_MODELS = [
  "grok-3-beta",
  "grok-3-fast-beta",  // Adicione ou remova modelos conforme necessário
];


// Configuração dos modelos Gemini (em ordem de preferência)
const GEMINI_MODELS = [
  "gemini-2.0-flash",
  "gemini-1.0-pro",
  "gemini-pro"
];

// Configuração dos modelos DeepSeek (em ordem de preferência)
const DEEPSEEK_MODELS = [
  "deepseek-reasoner",
  "deepseek-chat"  // Adicione ou remova modelos conforme necessário
];

// Configuração dos modelos Maritaca (em ordem de preferência)
const MARITACA_MODELS = [
  "sabia-3",  // Adicione ou remova modelos conforme necessário
  "sabiazinho-3",
];

//------------------------------------------------------------------------
// FIM DA CONFIGURAÇÃO GLOBAL
//------------------------------------------------------------------------

// Inicializar o servidor Express
const app = express();
const port = process.env.PORT || 3000;

// Configuração de middleware
app.use(cors());
app.use(express.json());
app.use('/uploads', express.static(path.join(__dirname, 'uploads')));
app.use('/responses', express.static(path.join(__dirname, 'responses')));

// Configurações de upload de arquivos
const storage = multer.diskStorage({
    destination: (req, file, cb) => {
      cb(null, 'uploads/');
    },
    filename: (req, file, cb) => {
      cb(null, Date.now() + path.extname(file.originalname));
    }
  });

  const upload = multer({
    storage: storage,
    limits: { 
      fileSize: ocrConfig.imagePreprocessing.maxImageSizeMB * 1024 * 1024 
    },
    fileFilter: (req, file, cb) => {
      if (!file.mimetype.startsWith('image/')) {
        return cb(new Error('Apenas imagens são permitidas'));
      }
      cb(null, true);
    }
  });

// Inicializar clientes de IA
let anthropic = null;
let openai = null;
let genAI = null;
let deepseek = null;
let maritaca = null;
let xAi = null;

// Inicializar os clientes apenas se as chaves estiverem disponíveis
try {
  if (process.env.ANTHROPIC_API_KEY) {
    anthropic = new Anthropic({
      apiKey: process.env.ANTHROPIC_API_KEY,
    });
    console.log("Cliente Anthropic (Claude) inicializado com sucesso");
    
  } else {
    console.log("Chave de API do Anthropic não configurada");
  }
} catch (error) {
  console.error("Erro ao inicializar cliente Anthropic:", error.message);
}

try {
  if (process.env.OPENAI_API_KEY) {
    openai = new OpenAI({
      apiKey: process.env.OPENAI_API_KEY,
    });
    console.log("Cliente OpenAI (GPT) inicializado com sucesso");
  } else {
    console.log("Chave de API do OpenAI não configurada");
  }
} catch (error) {
  console.error("Erro ao inicializar cliente OpenAI:", error.message);
}

try{
  if(process.env.XAI_API_KEY){
    xAi = new OpenAI({
      baseURL: 'https://api.x.ai/v1',
      apiKey: process.env.XAI_API_KEY,
    });
    console.log("Cliente XAI inicializado com sucesso");
} else{
    console.log("Chave de API do XAI não configurada");
  }
} catch (error) {
  console.error("Erro ao inicializar cliente XAI:", error.message);
}

try {
  if (process.env.GEMINI_API_KEY) {
    genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
    console.log("Cliente Google Generative AI (Gemini) inicializado com sucesso");
  } else {
    console.log("Chave de API do Gemini não configurada");
  }
} catch (error) {
  console.error("Erro ao inicializar cliente Gemini:", error.message);
}

try {
  if (process.env.DEEPSEEK_API_KEY) {
    deepseek = new OpenAI({
      baseURL: 'https://api.deepseek.com',
      apiKey: process.env.DEEPSEEK_API_KEY,
    });
    console.log("Cliente DeepSeek inicializado com sucesso");
  } else {
    console.log("Chave de API do DeepSeek não configurada");
  }
} catch (error) {
  console.error("Erro ao inicializar cliente DeepSeek:", error.message);
}

try {
  if (process.env.MARITACA_API_KEY) {
    maritaca = new OpenAI({
      baseURL: 'https://chat.maritaca.ai/api',
      apiKey: process.env.MARITACA_API_KEY,
    });
    console.log("Cliente Maritaca inicializado com sucesso");
  } else {
    console.log("Chave de API do Maritaca não configurada");
  }
} catch (error) {
  console.error("Erro ao inicializar cliente Maritaca:", error.message);
}

// Garantir que as pastas necessárias existam
const ensureDirExists = (dirPath) => {
  if (!fs.existsSync(dirPath)) {
    fs.mkdirSync(dirPath, { recursive: true });
  }
};

ensureDirExists(path.join(__dirname, 'uploads'));
ensureDirExists(path.join(__dirname, 'responses'));

const TIMEOUT_MS = 20000; // 20 segundos
const MAX_RETRIES = 2;
const INITIAL_RETRY_DELAY = 1000; // 1 segundo

/**
 * Adiciona um timeout a qualquer Promise
 * @param {Promise} promise - A promise para adicionar timeout
 * @param {number} timeoutMs - Tempo em ms para o timeout
 * @param {string} errorMessage - Mensagem de erro para o timeout
 * @returns {Promise} - Nova promise com timeout
 */
function withTimeout(promise, timeoutMs = TIMEOUT_MS, errorMessage = 'Operação excedeu o tempo limite') {
  const timeoutPromise = new Promise((_, reject) => {
    const id = setTimeout(() => {
      clearTimeout(id);
      reject(new Error(errorMessage));
    }, timeoutMs);
  });

  return Promise.race([promise, timeoutPromise]);
}

/**
 * Processa a resposta de texto para extrair VERDADEIRO ou FALSO
 * @param {string} text - Texto de resposta a ser processado
 * @returns {string|null} - "VERDADEIRO", "FALSO" ou o texto original
 */
function processResponse(text) {
  if (!text) return null;
  
  const normalizedText = text.trim();
  
  // Verificar padrões em português
  if (normalizedText.includes('VERDADEIRO')) return 'VERDADEIRO';
  if (normalizedText.includes('FALSO')) return 'FALSO';
  
  // Verificar padrões em inglês
  if (normalizedText.includes('TRUE')) return 'VERDADEIRO';
  if (normalizedText.includes('FALSE')) return 'FALSO';
  
  // Verificar abreviações
  if (normalizedText === 'V' || normalizedText.includes(' V ')) return 'VERDADEIRO';
  if (normalizedText === 'F' || normalizedText.includes(' F ')) return 'FALSO';
  
  // Se não conseguiu determinar, retornar o texto original
  return normalizedText;
}

/**
 * Cria um prompt melhorado para avaliação de verdadeiro/falso
 * @param {string} question - Texto da questão
 * @param {string|number} itemNumber - Número do item
 * @returns {string} - Prompt formatado
 */
function createPrompt(question, itemNumber) {
  return `
${question}

INSTRUÇÕES IMPORTANTES:
- Avalie se o item ${itemNumber} é VERDADEIRO ou FALSO com base no texto acima
- Responda APENAS com "VERDADEIRO" ou "FALSO" (em maiúsculas)
- NÃO forneça explicações ou justificativas
- Seja direto e objetivo
`;
}

/**
 * Função genérica para consultar modelos de IA
 * @param {Object} options - Opções para a consulta
 * @returns {Promise<string|null>} - Texto de resposta ou null em caso de falha
 */
async function askModel(options) {
  const { 
    client, 
    clientName, 
    models, 
    question, 
    itemNumber,
    makeRequest 
  } = options;

  try {
    if (!client) {
      console.log(`Cliente do ${clientName} não está configurado`);
      return null;
    }

    const prompt = createPrompt(question, itemNumber);
    
    // Tentar cada modelo da lista
    for (const modelName of models) {
      let retryCount = 0;
      let retryDelay = INITIAL_RETRY_DELAY;
      
      while (retryCount <= MAX_RETRIES) {
        try {
          console.log(`Consultando ${clientName} usando modelo ${modelName}` +
                     (retryCount > 0 ? ` (tentativa ${retryCount+1}/${MAX_RETRIES+1})` : ''));
          
          // Fazer a requisição com timeout
          const result = await withTimeout(
            makeRequest(client, modelName, prompt),
            TIMEOUT_MS,
            `Timeout de ${TIMEOUT_MS/1000}s excedido ao consultar o modelo ${modelName}`
          );
          
          return processResponse(result);
          
        } catch (retryError) {
          // Verificar se é um erro de timeout
          const isTimeout = retryError.message?.includes('Timeout de');
          
          if (isTimeout) {
            console.log(`${retryError.message}. Tentando próximo modelo...`);
            break; // Sai do loop while para tentar o próximo modelo
          }
          
          // Verificar se é erro de modelo não encontrado
          const isModelNotFound = 
            retryError.message?.includes('model not found') || 
            retryError.message?.includes('does not exist') ||
            retryError.message?.includes('not supported');
          
          // Se for erro de modelo, passamos para o próximo modelo
          if (isModelNotFound) {
            console.log(`Modelo ${modelName} não encontrado ou não suportado, tentando próximo modelo...`);
            break; // Sai do loop while para tentar o próximo modelo
          }
          
          // Verificar se é o último retry
          if (retryCount >= MAX_RETRIES) {
            console.log(`Esgotado máximo de tentativas para ${modelName}, tentando próximo modelo...`);
            break; // Sai do loop while para tentar o próximo modelo
          }
          
          // Calcular atraso para o próximo retry (backoff exponencial)
          console.log(`Erro ao consultar modelo. Aguardando ${retryDelay/1000} segundos para retry...`);
          await new Promise(resolve => setTimeout(resolve, retryDelay));
          
          // Aumentar o contador e o atraso para o próximo retry
          retryCount++;
          retryDelay *= 2; // Backoff exponencial
        }
      }
    }
    
    // Se chegou aqui, é porque todos os modelos falharam
    console.error(`Todos os modelos ${clientName} falharam`);
    return null;
  } catch (error) {
    console.error(`Erro ao consultar ${clientName}:`, error);
    return null;
  }
}

// Implementação específica para o Claude (Anthropic)
async function askClaude(question, itemNumber) {
  // Verificar se podemos usar a Messages API
  if (!anthropic || typeof anthropic.messages !== 'object' || typeof anthropic.messages.create !== 'function') {
    console.error('A versão da biblioteca @anthropic-ai/sdk não suporta a Messages API necessária para os modelos Claude 3.x');
    return null;
  }
  
  // Função para fazer a requisição ao Claude
  const makeClaudeRequest = async (client, modelName, prompt) => {
    const response = await client.messages.create({
      model: modelName,
      max_tokens: 50,
      messages: [
        { role: 'user', content: prompt }
      ]
    });
    
    return response.content[0].text;
  };
  
  return askModel({
    client: anthropic,
    clientName: 'Claude', 
    models: CLAUDE_MODELS,
    question,
    itemNumber,
    makeRequest: makeClaudeRequest
  });
}

// Implementação específica para o GPT (OpenAI)
async function askGPT(question, itemNumber) {
  // Função para fazer a requisição ao GPT
  const makeGPTRequest = async (client, modelName, prompt) => {
    const response = await client.chat.completions.create({
      model: modelName,
      messages: [
        { role: 'system', content: 'You are an expert in public tenders from the CEBRASP examination board' },
        { role: 'user', content: prompt }
      ],
      max_tokens: 50,
      temperature: 0.1
    });
    
    return response.choices[0].message.content;
  };
  
  return askModel({
    client: openai,
    clientName: 'GPT', 
    models: GPT_MODELS,
    question,
    itemNumber,
    makeRequest: makeGPTRequest
  });
}

// Implementação específica para o Gemini (Google)
async function askGemini(question, itemNumber) {
  // Função para fazer a requisição ao Gemini
  const makeGeminiRequest = async (client, modelName, prompt) => {
    const model = client.getGenerativeModel({ model: modelName });
    const result = await model.generateContent(prompt);
    const response = await result.response;
    
    return response.text();
  };
  
  return askModel({
    client: genAI,
    clientName: 'Gemini', 
    models: GEMINI_MODELS,
    question,
    itemNumber,
    makeRequest: makeGeminiRequest
  });
}

// Implementação específica para o DeepSeek
async function askDeepSeek(question, itemNumber) {
  // Função para fazer a requisição ao DeepSeek
  const makeDeepSeekRequest = async (client, modelName, prompt) => {
    const response = await client.chat.completions.create({
      model: modelName,
      messages: [
        { role: 'system', content: 'You are an expert in public tenders from the CEBRASP examination board.' },
        { role: 'user', content: prompt }
      ],
      temperature: 0.1,
      max_tokens: 50
    });
    
    return response.choices[0].message.content;
  };
  
  return askModel({
    client: deepseek,
    clientName: 'DeepSeek', 
    models: DEEPSEEK_MODELS,
    question,
    itemNumber,
    makeRequest: makeDeepSeekRequest
  });
}

// Implementação específica para o Maritaca
async function askMaritaca(question, itemNumber) {
  // Função para fazer a requisição ao Maritaca
  const makeMaritacaRequest = async (client, modelName, prompt) => {
    const response = await client.chat.completions.create({
      model: modelName,
      messages: [
        { role: 'user', content: prompt }
      ],
      temperature: 0.1,
      max_tokens: 50
    });
    
    return response.choices[0].message.content;
  };
  
  return askModel({
    client: maritaca,
    clientName: 'Maritaca', 
    models: MARITACA_MODELS,
    question,
    itemNumber,
    makeRequest: makeMaritacaRequest
  });
}

async function askXAI(question, itemNumber) {
  // Função para fazer a requisição ao XAI
  const makeXAIRequest = async (client, modelName, prompt) => {
    const response = await client.chat.completions.create({
      model: modelName,
      messages: [
        { role: 'system', content: 'You are an expert in public tenders from the CEBRASP examination board.' },
        { role: 'user', content: prompt }
      ],
      temperature: 0.1,
      max_tokens: 50
    });
    
    return response.choices[0].message.content;
  };
  
  return askModel({
    client: xAi,
    clientName: 'XAI', 
    models: XAI_MODELS,
    question,
    itemNumber,
    makeRequest: makeXAIRequest
  });
}

function findCommonResponse(claudeResponse, gptResponse, geminiResponse, deepseekResponse, maritacaResponse, xaiResponse) {
  // Se alguma resposta estiver faltando, retorna as disponíveis
  const responses = [claudeResponse, gptResponse, geminiResponse, deepseekResponse, maritacaResponse, xaiResponse].filter(r => r && r.trim() !== '');
  
  if (responses.length === 0) {
    console.log('Nenhuma resposta válida obtida de nenhum modelo');
    return null;
  }
  
  if (responses.length === 1) {
    console.log('Apenas uma resposta válida disponível:', responses[0]);
    return responses[0];
  }
  
  // Normalizar respostas para apenas VERDADEIRO ou FALSO
  const normalizeResponse = (response) => {
    if (!response) return null;
    
    // Verificar padrões de resposta
    if (response.includes('VERDADEIRO') || 
        response.includes('TRUE') || 
        response === 'V' || 
        response.includes(' V ')) {
      return 'VERDADEIRO';
    }
    
    if (response.includes('FALSO') || 
        response.includes('FALSE') || 
        response === 'F' || 
        response.includes(' F ')) {
      return 'FALSO';
    }
    
    // Se não conseguiu normalizar, retornar null
    return null;
  };
  
  // Definir pesos para cada modelo (usados no desempate)
  const MODEL_WEIGHTS = {
    'claude': 5,
    'gemini': 6,
    'xai': 5,
    'gpt': 4,
    'deepseek': 3,
    'maritaca': 3
  };
  
  // Normalizar e criar array com respostas, modelo e peso
  const allResponsesWithModels = [
    { model: 'claude', response: normalizeResponse(claudeResponse), weight: MODEL_WEIGHTS['claude'] },
    { model: 'gpt', response: normalizeResponse(gptResponse), weight: MODEL_WEIGHTS['gpt'] },
    { model: 'gemini', response: normalizeResponse(geminiResponse), weight: MODEL_WEIGHTS['gemini'] },
    { model: 'deepseek', response: normalizeResponse(deepseekResponse), weight: MODEL_WEIGHTS['deepseek'] },
    { model: 'maritaca', response: normalizeResponse(maritacaResponse), weight: MODEL_WEIGHTS['maritaca'] },
    { model: 'xai', response: normalizeResponse(xaiResponse), weight: MODEL_WEIGHTS['xai'] }
  ].filter(item => item.response !== null);
  
  if (allResponsesWithModels.length === 0) {
    console.log('Nenhuma resposta pôde ser normalizada');
    return responses[0]; // Retornar a primeira resposta original
  }
  
  // Contar votos para VERDADEIRO e FALSO
  let verdadeiroCount = 0;
  let falsoCount = 0;
  let verdadeiroWeight = 0;
  let falsoWeight = 0;
  let verdadeiroModels = [];
  let falsoModels = [];
  
  allResponsesWithModels.forEach(item => {
    if (item.response === 'VERDADEIRO') {
      verdadeiroCount++;
      verdadeiroWeight += item.weight;
      verdadeiroModels.push(item.model);
    } else if (item.response === 'FALSO') {
      falsoCount++;
      falsoWeight += item.weight;
      falsoModels.push(item.model);
    }
  });
  
  console.log(`Contagem de votos - VERDADEIRO: ${verdadeiroCount} (peso ${verdadeiroWeight}), FALSO: ${falsoCount} (peso ${falsoWeight})`);
  
  // Regras de decisão
  
  // 1. Se um dos lados tem 3+ votos, esse vence
  if (verdadeiroCount >= 3 && verdadeiroCount > falsoCount) {
    console.log(`Consenso forte: VERDADEIRO com ${verdadeiroCount} votos de ${verdadeiroModels.join(', ')}`);
    return 'VERDADEIRO';
  }
  
  if (falsoCount >= 3 && falsoCount > verdadeiroCount) {
    console.log(`Consenso forte: FALSO com ${falsoCount} votos de ${falsoModels.join(', ')}`);
    return 'FALSO';
  }
  
  // 2. Se os dois lados têm votos iguais, desempatar por peso
  if (verdadeiroCount === falsoCount) {
    if (verdadeiroWeight > falsoWeight) {
      console.log(`Empate em votos, desempate por peso: VERDADEIRO vence (peso ${verdadeiroWeight} vs ${falsoWeight})`);
      return 'VERDADEIRO';
    } else {
      console.log(`Empate em votos, desempate por peso: FALSO vence (peso ${falsoWeight} vs ${verdadeiroWeight})`);
      return 'FALSO';
    }
  }
  
  // 3. Verificar se Claude e Gemini concordam (são os modelos mais confiáveis)
  const claudeResponse2 = allResponsesWithModels.find(item => item.model === 'claude')?.response;
  const geminiResponse2 = allResponsesWithModels.find(item => item.model === 'gemini')?.response;
  const xResponse2 = allResponsesWithModels.find(item => item.model === 'xai')?.response;
  
  if (claudeResponse2 && xResponse2 && geminiResponse2 && claudeResponse2 === geminiResponse2 && xResponse2 === geminiResponse2 && xResponse2 === claudeResponse2) {
    console.log(`Consenso forte: Claude, Gemini e Grok concordam em ${claudeResponse2}`);
    return claudeResponse2;
  }
  
  // 4. Se chegou aqui, retorna o lado com mais votos
  if (verdadeiroCount > falsoCount) {
    console.log(`VERDADEIRO vence por ${verdadeiroCount} a ${falsoCount}`);
    return 'VERDADEIRO';
  } else {
    console.log(`FALSO vence por ${falsoCount} a ${verdadeiroCount}`);
    return 'FALSO';
  }
}

// Função auxiliar para calcular similaridade entre strings
function calculateSimilarity(str1, str2) {
  const words1 = str1.toLowerCase().split(/\s+/);
  const words2 = str2.toLowerCase().split(/\s+/);
  
  const set1 = new Set(words1);
  const set2 = new Set(words2);
  
  const intersection = new Set([...set1].filter(x => set2.has(x)));
  const union = new Set([...set1, ...set2]);
  
  return intersection.size / union.size;
}

// Função para salvar a resposta em um arquivo de texto
async function saveResponseToFile(text) {
  try {
    // Garantir que a pasta 'responses' exista
    const responsesDir = path.join(__dirname, 'responses');
    if (!fs.existsSync(responsesDir)) {
      fs.mkdirSync(responsesDir, { recursive: true });
    }
    
    // Gerar um nome de arquivo baseado no timestamp
    const filename = `response-${Date.now()}.txt`;
    const filePath = path.join(responsesDir, filename);
    
    // Escrever o texto no arquivo
    fs.writeFileSync(filePath, text);
    console.log(`Resposta salva em: ${filePath}`);
    
    // Retornar o caminho relativo do arquivo
    return `responses/${filename}`;
  } catch (error) {
    console.error('Erro ao salvar resposta em arquivo:', error);
    throw new Error('Falha ao salvar resposta');
  }
}


/**
 * @swagger
 * /api/analyze:
 *   post:
 *     summary: Analisa uma imagem de questão
 *     description: |
 *       Recebe uma imagem contendo uma questão de múltipla escolha, extrai o texto usando OCR,
 *       e consulta múltiplos modelos de IA para determinar a alternativa correta.
 *       
 *       O sistema consulta os seguintes modelos (quando disponíveis):
 *       - Claude (Anthropic)
 *       - GPT (OpenAI)
 *       - Grok (xAI)
 *       - Gemini (Google)
 *       - DeepSeek
 *       - Maritaca (Sabiá)
 *       
 *       A API retorna a resposta consensual entre os modelos, além das respostas individuais.
 *     tags: [Análise]
 *     consumes:
 *       - multipart/form-data
 *     requestBody:
 *       required: true
 *       content:
 *         multipart/form-data:
 *           schema:
 *             type: object
 *             properties:
 *               image:
 *                 type: string
 *                 format: binary
 *                 description: Imagem da questão a ser analisada (PNG, JPG, JPEG)
 */
app.post('/api/analyze', upload.single('image'), async (req, res) => {
  try {
    // Extrair texto e itens da imagem
    const extractedData = await extractTextFromImage(req.file.path);
    
    console.log(`Analisando ${extractedData.itens.length} itens...`);
    
    // Analisar cada item com múltiplos modelos de IA
    const itensAnalysed = await Promise.all(extractedData.itens.map(async (item) => {
      console.log(`Analisando item ${item.numero}...`);

      // Preparar o texto completo para análise (texto principal + item)
      const fullText = `${extractedData.texto_principal}\n\n${item.afirmacao}`;

      // Consultar todos os modelos de IA em paralelo
      const [claudeResponse, gptResponse, geminiResponse, deepseekResponse, maritacaResponse, xaiResponse] = await Promise.all([
        askClaude(fullText, item.numero),
        askGPT(fullText, item.numero),
        askGemini(fullText, item.numero),
        askDeepSeek(fullText, item.numero),
        askMaritaca(fullText, item.numero),
        askXAI(fullText, item.numero)
      ]);

      // Encontrar resposta consensual
      const commonResponse = findCommonResponse(
        claudeResponse, 
        gptResponse, 
        geminiResponse, 
        deepseekResponse, 
        maritacaResponse,
        xaiResponse
      );

      // Solicitar uma justificativa ao Claude (geralmente é o mais preciso em explicações)
//       let justificativa = '';
//       try {
//         if (anthropic) {
//           const justPrompt = `
// ${extractedData.texto_principal}

// Item ${item.numero}: ${item.afirmacao}

// Este item foi avaliado como ${commonResponse}.

// Por favor, forneça uma justificativa concisa para esta resposta, explicando por que o item é ${commonResponse} com base no texto.
// Mantenha a explicação objetiva e direta, com no máximo 2-3 frases.
// `;

//           for (const modelName of CLAUDE_MODELS) {
//             try {
//               const justResponse = await anthropic.messages.create({
//                 model: modelName,
//                 max_tokens: 150,
//                 messages: [
//                   { role: 'user', content: justPrompt }
//                 ]
//               });
              
//               justificativa = justResponse.content[0].text.trim();
//               break; // Se conseguiu, sai do loop
//             } catch (err) {
//               console.log(`Erro ao obter justificativa do modelo ${modelName}, tentando outro...`);
//             }
//           }
//         }
//       } catch (error) {
//         console.error('Erro ao obter justificativa:', error);
//         justificativa = 'Não foi possível gerar uma justificativa.';
//       }

      // if (!justificativa) {
      //   justificativa = 'Não foi possível gerar uma justificativa.';
      // }

      return {
        ...item,
        resposta: commonResponse,
        //justificativa: justificativa,
        respostas_modelos: {
          claude: claudeResponse,
          gpt: gptResponse,
          gemini: geminiResponse,
          deepseek: deepseekResponse,
          maritaca: maritacaResponse,
          xai: xaiResponse
        }
      };
    }));

    // Criar objeto com resultados completos
    const analysisData = {
      texto_principal: extractedData.texto_principal,
      itens: itensAnalysed
    };
    
    // Formatar as respostas
    const formattedResult = formatAnalysisResult(analysisData);
    
    // Salvar resposta em arquivo
    let responseUrl = null;
    if (formattedResult) {
      responseUrl = await saveResponseToFile(formattedResult);
    }
    
    // Enviar resposta JSON
    res.json({
      success: true,
      textoPrincipal: analysisData.texto_principal,
      itens: analysisData.itens,
      responseUrl: responseUrl,
      formattedResult: formattedResult
    });
  
  } catch (error) {
    console.error('Erro completo:', error);
    res.status(500).json({ 
      error: error.message || 'Erro interno no servidor'
    });
  }
});

/**
 * @swagger
 * /:
 *   get:
 *     summary: Verifica status da API
 *     description: Retorna um status simples para confirmar que a API está online e informações básicas
 *     tags: [Status]
 *     responses:
 *       200:
 *         description: API está funcionando
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/StatusResponse'
 */
app.get('/', (req, res) => {
  const startTime = process.uptime();
  const packageJson = require('./package.json');
  
  res.json({
    status: 'online',
    message: 'API de análise de imagens com IA está funcionando',
    version: packageJson.version,
    uptime: Math.floor(startTime)
  });
});

app.use('/api-docs', swaggerUi.serve, swaggerUi.setup(swaggerSpec, {
  explorer: true,
  customCss: '.swagger-ui .topbar { display: none }',
  customSiteTitle: "API de Análise de Questões - Documentação",
  customfavIcon: ""
}));

// Iniciar o servidor
app.listen(port, () => {
  console.log(`Servidor rodando na porta ${port}`);
  console.log(`Documentação Swagger disponível em http://localhost:${port}/api-docs`);
});

module.exports = app;