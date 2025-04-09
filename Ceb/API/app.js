// app.js - Servidor principal da API
const express = require('express');
const multer = require('multer');
const cors = require('cors');
const { Anthropic } = require('@anthropic-ai/sdk');
const { OpenAI } = require('openai');
const { GoogleGenerativeAI } = require('@google/generative-ai');
const fs = require('fs');
const path = require('path');
const dotenv = require('dotenv');

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
  "claude-3-opus-20240229",      // Claude 3 Opus (mais potente, mas mais antigo)
  "claude-3-5-sonnet-20240620",  // Claude 3.5 Sonnet (versão antiga)
  "claude-3-haiku-20240307"      // Claude 3 Haiku (versão antiga)
];

// Configuração dos modelos GPT (em ordem de preferência)
const GPT_MODELS = [
  "gpt-3.5-turbo",
  "gpt-3.5-turbo-instruct",
  "text-davinci-003"
];

// Configuração dos modelos Gemini (em ordem de preferência)
const GEMINI_MODELS = [
  "gemini-2.0-flash",
  "gemini-1.0-pro",
  "gemini-pro"
];

// Configuração dos modelos DeepSeek (em ordem de preferência)
const DEEPSEEK_MODELS = [
  "deepseek-chat",
  "deepseek-coder"  // Adicione ou remova modelos conforme necessário
];

// Configuração dos modelos Maritaca (em ordem de preferência)
const MARITACA_MODELS = [
  "sabia-3",
  "sabia-2"  // Adicione ou remova modelos conforme necessário
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

// Inicializar os clientes apenas se as chaves estiverem disponíveis
try {
  if (process.env.ANTHROPIC_API_KEY) {
    anthropic = new Anthropic({
      apiKey: process.env.ANTHROPIC_API_KEY,
    });
    console.log("Cliente Anthropic (Claude) inicializado com sucesso");

    
    // Verificar qual tipo de API está disponível
    if (anthropic.messages && anthropic.messages.create) {
      console.log("API do Claude usando messages.create");
    } else if (anthropic.completions && anthropic.completions.create) {
      console.log("API do Claude usando completions.create");
    } else if (anthropic.create) {
      console.log("API do Claude usando create diretamente");
    } else {
      console.log("AVISO: Interface da API do Claude não reconhecida");
    }
    
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

// Função para obter resposta do Claude
async function askClaude(question, itemNumber) {
  try {
    if (!anthropic) {
      console.log('Cliente do Claude não está configurado');
      return null;
    }
    
    // Modificar o prompt para obter apenas a resposta verdadeiro/falso
    const enhancedPrompt = `
${question}

INSTRUÇÕES IMPORTANTES:
- Avalie se o item ${itemNumber} é VERDADEIRO ou FALSO com base no texto acima
- Responda APENAS com "VERDADEIRO" ou "FALSO" (em maiúsculas)
- NÃO forneça explicações ou justificativas
- Seja direto e objetivo
`;
    
    // Verificar se temos acesso à Messages API (única compatível com Claude 3.x)
    if (typeof anthropic.messages === 'object' && typeof anthropic.messages.create === 'function') {
      // Configuração de retry
      const MAX_RETRIES = 2;
      const INITIAL_RETRY_DELAY = 1000; // 1 segundo
      
      // Tentar cada modelo sequencialmente
      for (const modelName of CLAUDE_MODELS) {
        let retryCount = 0;
        let retryDelay = INITIAL_RETRY_DELAY;
        
        while (retryCount <= MAX_RETRIES) {
          try {
            console.log(`Consultando Claude usando modelo ${modelName}` + 
                        (retryCount > 0 ? ` (tentativa ${retryCount+1}/${MAX_RETRIES+1})` : ''));
            
            const response = await anthropic.messages.create({
              model: modelName,
              max_tokens: 50,
              messages: [
                { role: 'user', content: enhancedPrompt }
              ]
            });
            
            const responseText = response.content[0].text.trim();
            
            // Processar a resposta para extrair apenas VERDADEIRO ou FALSO
            if (responseText.includes('VERDADEIRO')) {
              return 'VERDADEIRO';
            } else if (responseText.includes('FALSO')) {
              return 'FALSO';
            }
            
            // Se não encontrou um padrão claro, verificar por true/false em inglês
            if (responseText.includes('TRUE')) {
              return 'VERDADEIRO';
            } else if (responseText.includes('FALSE')) {
              return 'FALSO';
            }
            
            // Se ainda não encontrou, verificar por V/F
            if (responseText.includes(' V ') || responseText === 'V') {
              return 'VERDADEIRO';
            } else if (responseText.includes(' F ') || responseText === 'F') {
              return 'FALSO';
            }
            
            // Se ainda não conseguimos extrair um padrão conhecido, retornar o texto original
            return responseText;
            
          } catch (retryError) {
            // Verificar tipos de erros
            const isOverloaded = 
              retryError.status === 529 || 
              (retryError.error?.error?.type === 'overloaded_error') ||
              (retryError.headers && retryError.headers['x-should-retry'] === 'true');
            
            const isModelNotFound = 
              retryError.status === 404 ||
              retryError.message?.includes('model not found') ||
              retryError.message?.includes('does not exist') ||
              retryError.message?.includes('not supported');
            
            // Se o modelo não existir, pular para o próximo
            if (isModelNotFound) {
              console.log(`Modelo ${modelName} não encontrado ou não suportado, tentando próximo modelo...`);
              break; // Sai do loop while para tentar o próximo modelo
            }
            
            // Se for o último retry ou não for um erro de sobrecarga, tentar próximo modelo
            if (retryCount >= MAX_RETRIES || !isOverloaded) {
              console.log(`Erro com modelo ${modelName}, tentando próximo modelo...`);
              break; // Sai do loop while para tentar o próximo modelo
            }
            
            // Calcular atraso para o próximo retry (backoff exponencial)
            console.log(`Servidor do Claude sobrecarregado. Aguardando ${retryDelay/1000} segundos para retry...`);
            await new Promise(resolve => setTimeout(resolve, retryDelay));
            
            // Aumentar o contador e o atraso para o próximo retry
            retryCount++;
            retryDelay *= 2; // Backoff exponencial
          }
        }
      }
      
      // Se chegou aqui, é porque todos os modelos falharam
      console.error('Todos os modelos Claude falharam');
      return null;
    } else {
      // Biblioteca desatualizada
      console.error('A versão da sua biblioteca @anthropic-ai/sdk não suporta a Messages API necessária para os modelos Claude 3.x');
      return null;
    }
  } catch (error) {
    console.error('Erro ao consultar Claude:', error);
    return null;
  }
}

// Função para obter resposta do GPT
async function askGPT(question, itemNumber) {
  try {
    if (!openai) {
      console.log('Cliente do GPT não está configurado');
      return null;
    }
    
    // Prompt melhorado com instruções mais claras
    const enhancedPrompt = `
${question}

INSTRUÇÕES IMPORTANTES:
- Avalie se o item ${itemNumber} é VERDADEIRO ou FALSO com base no texto acima
- Responda APENAS com "VERDADEIRO" ou "FALSO" (em maiúsculas)
- NÃO forneça explicações ou justificativas
- Seja direto e objetivo
`;
    
    // Configuração de retry
    const MAX_RETRIES = 2;
    const INITIAL_RETRY_DELAY = 1000; // 1 segundo
    
    let retryCount = 0;
    let retryDelay = INITIAL_RETRY_DELAY;
    
    // Usar a lista de modelos da configuração global
    const models = GPT_MODELS;
    
    // Tentar cada modelo até obter sucesso
    for (const model of models) {
      retryCount = 0;
      
      while (retryCount <= MAX_RETRIES) {
        try {
          console.log(`Consultando GPT usando modelo ${model}` + 
                     (retryCount > 0 ? ` (tentativa ${retryCount+1}/${MAX_RETRIES+1})` : ''));
          
          const response = await openai.chat.completions.create({
            model: model,
            messages: [
              { role: 'system', content: 'Você é um assistente especializado em responder questões de verdadeiro ou falso com extrema precisão e concisão. Siga EXATAMENTE o formato solicitado.' },
              { role: 'user', content: enhancedPrompt }
            ],
            max_tokens: 50,
            temperature: 0.1
          });
          
          const responseText = response.choices[0].message.content.trim();
          
          // Processar a resposta para extrair apenas VERDADEIRO ou FALSO
          if (responseText.includes('VERDADEIRO')) {
            return 'VERDADEIRO';
          } else if (responseText.includes('FALSO')) {
            return 'FALSO';
          }
          
          // Se não encontrou um padrão claro, verificar por true/false em inglês
          if (responseText.includes('TRUE')) {
            return 'VERDADEIRO';
          } else if (responseText.includes('FALSE')) {
            return 'FALSO';
          }
          
          // Se ainda não encontrou, verificar por V/F
          if (responseText.includes(' V ') || responseText === 'V') {
            return 'VERDADEIRO';
          } else if (responseText.includes(' F ') || responseText === 'F') {
            return 'FALSO';
          }
          
          // Se ainda não conseguimos extrair um padrão conhecido
          console.log(`Formato de resposta não reconhecido: "${responseText}"`);
          break;
          
        } catch (retryError) {
          // Verificar se é um erro de quota ou rate limit
          const isRetryable = 
            retryError.status === 429 ||
            retryError.code === 'insufficient_quota' ||
            retryError.message?.includes('rate limit') ||
            retryError.message?.includes('quota');
          
          // Verificar se o erro é devido ao modelo não existir
          const isModelNotFound = 
            retryError.message?.includes('model not found') ||
            retryError.message?.includes('does not exist');
          
          // Se for erro de modelo, passamos para o próximo modelo
          if (isModelNotFound) {
            console.log(`Modelo ${model} não encontrado, tentando próximo modelo...`);
            break; // Sai do loop while para tentar o próximo modelo
          }
          
          // Se for o último retry ou não for um erro retryable, passamos para o próximo modelo
          if (retryCount >= MAX_RETRIES || !isRetryable) {
            console.log(`Erro com modelo ${model}, tentando próximo modelo...`);
            break; // Sai do loop while para tentar o próximo modelo
          }
          
          // Calcular atraso para o próximo retry (backoff exponencial)
          console.log(`Limite de taxa excedido para ${model}. Aguardando ${retryDelay/1000} segundos para retry...`);
          await new Promise(resolve => setTimeout(resolve, retryDelay));
          
          // Aumentar o contador e o atraso para o próximo retry
          retryCount++;
          retryDelay *= 2; // Backoff exponencial
        }
      }
    }
    
    // Se chegou aqui, é porque nenhum modelo funcionou
    console.error('Todos os modelos GPT falharam');
    return null;
  } catch (error) {
    console.error('Erro ao consultar GPT:', error);
    return null;
  }
}

// Função para obter resposta do Gemini
async function askGemini(question, itemNumber) {
  try {
    if (!genAI) {
      console.log('Cliente do Gemini não está configurado');
      return null;
    }
    
    // Prompt para obter apenas a resposta verdadeiro/falso
    const enhancedPrompt = `
${question}

INSTRUÇÕES IMPORTANTES:
- Avalie se o item ${itemNumber} é VERDADEIRO ou FALSO com base no texto acima
- Responda APENAS com "VERDADEIRO" ou "FALSO" (em maiúsculas)
- NÃO forneça explicações ou justificativas
- Seja direto e objetivo
`;
    
    // Configuração de retry
    const MAX_RETRIES = 2;
    const INITIAL_RETRY_DELAY = 1000; // 1 segundo
    
    // Usar a lista de modelos da configuração global
    const modelOptions = GEMINI_MODELS;
    
    // Para cada modelo, tente com retries
    for (const modelName of modelOptions) {
      let retryCount = 0;
      let retryDelay = INITIAL_RETRY_DELAY;
      
      while (retryCount <= MAX_RETRIES) {
        try {
          console.log(`Consultando Gemini usando modelo ${modelName}` +
                     (retryCount > 0 ? ` (tentativa ${retryCount+1}/${MAX_RETRIES+1})` : ''));
          
          const model = genAI.getGenerativeModel({ model: modelName });
          const result = await model.generateContent(enhancedPrompt);
          const response = await result.response;
          const text = response.text().trim();
          
          // Processar a resposta para extrair apenas VERDADEIRO ou FALSO
          if (text.includes('VERDADEIRO')) {
            return 'VERDADEIRO';
          } else if (text.includes('FALSO')) {
            return 'FALSO';
          }
          
          // Se não encontrou um padrão claro, verificar por true/false em inglês
          if (text.includes('TRUE')) {
            return 'VERDADEIRO';
          } else if (text.includes('FALSE')) {
            return 'FALSO';
          }
          
          // Se ainda não encontrou, verificar por V/F
          if (text.includes(' V ') || text === 'V') {
            return 'VERDADEIRO';
          } else if (text.includes(' F ') || text === 'F') {
            return 'FALSO';
          }
          
          // Retornar o texto original se não encontrou nenhum padrão conhecido
          return text;
          
        } catch (retryError) {
          console.error(`Erro ao usar modelo ${modelName}:`, retryError.message);
          
          // Verificar se é o último retry
          if (retryCount >= MAX_RETRIES) {
            console.log(`Esgotado máximo de tentativas para ${modelName}, tentando próximo modelo...`);
            break;
          }
          
          // Calcular atraso para o próximo retry (backoff exponencial)
          console.log(`Aguardando ${retryDelay/1000} segundos para retry...`);
          await new Promise(resolve => setTimeout(resolve, retryDelay));
          
          // Aumentar o contador e o atraso para o próximo retry
          retryCount++;
          retryDelay *= 2; // Backoff exponencial
        }
      }
    }
    
    // Se chegou aqui, é porque todos os modelos falharam
    console.error('Todos os modelos Gemini falharam');
    return null;
    
  } catch (error) {
    console.error('Erro ao consultar Gemini:', error);
    return null;
  }
}

// Função para obter resposta do DeepSeek
async function askDeepSeek(question, itemNumber) {
  try {
    if (!deepseek) {
      console.log('Cliente do DeepSeek não está configurado');
      return null;
    }
    
    // Prompt para obter apenas a resposta verdadeiro/falso
    const enhancedPrompt = `
${question}

INSTRUÇÕES IMPORTANTES:
- Avalie se o item ${itemNumber} é VERDADEIRO ou FALSO com base no texto acima
- Responda APENAS com "VERDADEIRO" ou "FALSO" (em maiúsculas)
- NÃO forneça explicações ou justificativas
- Seja direto e objetivo
`;
    
    // Configuração de retry
    const MAX_RETRIES = 2;
    const INITIAL_RETRY_DELAY = 1000; // 1 segundo
    
    // Para cada modelo, tente com retries
    for (const modelName of DEEPSEEK_MODELS) {
      let retryCount = 0;
      let retryDelay = INITIAL_RETRY_DELAY;
      
      while (retryCount <= MAX_RETRIES) {
        try {
          console.log(`Consultando DeepSeek usando modelo ${modelName}` +
                     (retryCount > 0 ? ` (tentativa ${retryCount+1}/${MAX_RETRIES+1})` : ''));
          
          const response = await deepseek.chat.completions.create({
            model: modelName,
            messages: [
              { role: 'system', content: 'You are a helpful assistant.' },
              { role: 'user', content: enhancedPrompt }
            ],
            temperature: 0.3,
            max_tokens: 50
          });
          
          const responseText = response.choices[0].message.content.trim();
          
          // Processar a resposta para extrair apenas VERDADEIRO ou FALSO
          if (responseText.includes('VERDADEIRO')) {
            return 'VERDADEIRO';
          } else if (responseText.includes('FALSO')) {
            return 'FALSO';
          }
          
          // Se não encontrou um padrão claro, verificar por true/false em inglês
          if (responseText.includes('TRUE')) {
            return 'VERDADEIRO';
          } else if (responseText.includes('FALSE')) {
            return 'FALSO';
          }
          
          // Se ainda não encontrou, verificar por V/F
          if (responseText.includes(' V ') || responseText === 'V') {
            return 'VERDADEIRO';
          } else if (responseText.includes(' F ') || responseText === 'F') {
            return 'FALSO';
          }
          
          // Retornar o texto original se não encontrou nenhum padrão conhecido
          return responseText;
          
        } catch (retryError) {
          console.error(`Erro ao usar modelo ${modelName}:`, retryError.message);
          
          // Verificar se é erro de modelo não encontrado
          const isModelNotFound = 
            retryError.message?.includes('model not found') ||
            retryError.message?.includes('does not exist');
          
          // Se for erro de modelo, passamos para o próximo modelo
          if (isModelNotFound) {
            console.log(`Modelo ${modelName} não encontrado, tentando próximo modelo...`);
            break; // Sai do loop while para tentar o próximo modelo
          }
          
          // Verificar se é o último retry
          if (retryCount >= MAX_RETRIES) {
            console.log(`Esgotado máximo de tentativas para ${modelName}, tentando próximo modelo...`);
            break;
          }
          
          // Calcular atraso para o próximo retry (backoff exponencial)
          console.log(`Aguardando ${retryDelay/1000} segundos para retry...`);
          await new Promise(resolve => setTimeout(resolve, retryDelay));
          
          // Aumentar o contador e o atraso para o próximo retry
          retryCount++;
          retryDelay *= 2; // Backoff exponencial
        }
      }
    }
    
    // Se chegou aqui, é porque todos os modelos falharam
    console.error('Todos os modelos DeepSeek falharam');
    return null;
    
  } catch (error) {
    console.error('Erro ao consultar DeepSeek:', error);
    return null;
  }
}

// Função para obter resposta do Maritaca
async function askMaritaca(question, itemNumber) {
  try {
    if (!maritaca) {
      console.log('Cliente do Maritaca não está configurado');
      return null;
    }
    
    // Prompt para obter apenas a resposta verdadeiro/falso
    const enhancedPrompt = `
${question}

INSTRUÇÕES IMPORTANTES:
- Avalie se o item ${itemNumber} é VERDADEIRO ou FALSO com base no texto acima
- Responda APENAS com "VERDADEIRO" ou "FALSO" (em maiúsculas)
- NÃO forneça explicações ou justificativas
- Seja direto e objetivo
`;
    
    // Configuração de retry
    const MAX_RETRIES = 2;
    const INITIAL_RETRY_DELAY = 1000; // 1 segundo
    
    // Para cada modelo, tente com retries
    for (const modelName of MARITACA_MODELS) {
      let retryCount = 0;
      let retryDelay = INITIAL_RETRY_DELAY;
      
      while (retryCount <= MAX_RETRIES) {
        try {
          console.log(`Consultando Maritaca usando modelo ${modelName}` +
                     (retryCount > 0 ? ` (tentativa ${retryCount+1}/${MAX_RETRIES+1})` : ''));
          
          const response = await maritaca.chat.completions.create({
            model: modelName,
            messages: [
              { role: 'user', content: enhancedPrompt }
            ],
            temperature: 0.3,
            max_tokens: 50
          });
          
          const responseText = response.choices[0].message.content.trim();
          
          // Processar a resposta para extrair apenas VERDADEIRO ou FALSO
          if (responseText.includes('VERDADEIRO')) {
            return 'VERDADEIRO';
          } else if (responseText.includes('FALSO')) {
            return 'FALSO';
          }
          
          // Se não encontrou um padrão claro, verificar por true/false em inglês
          if (responseText.includes('TRUE')) {
            return 'VERDADEIRO';
          } else if (responseText.includes('FALSE')) {
            return 'FALSO';
          }
          
          // Se ainda não encontrou, verificar por V/F
          if (responseText.includes(' V ') || responseText === 'V') {
            return 'VERDADEIRO';
          } else if (responseText.includes(' F ') || responseText === 'F') {
            return 'FALSO';
          }
          
          // Retornar o texto original se não encontrou nenhum padrão conhecido
          return responseText;
          
        } catch (retryError) {
          console.error(`Erro ao usar modelo ${modelName}:`, retryError.message);
          
          // Verificar se é erro de modelo não encontrado
          const isModelNotFound = 
            retryError.message?.includes('model not found') ||
            retryError.message?.includes('does not exist');
          
          // Se for erro de modelo, passamos para o próximo modelo
          if (isModelNotFound) {
            console.log(`Modelo ${modelName} não encontrado, tentando próximo modelo...`);
            break; // Sai do loop while para tentar o próximo modelo
          }
          
          // Verificar se é o último retry
          if (retryCount >= MAX_RETRIES) {
            console.log(`Esgotado máximo de tentativas para ${modelName}, tentando próximo modelo...`);
            break;
          }
          
          // Calcular atraso para o próximo retry (backoff exponencial)
          console.log(`Aguardando ${retryDelay/1000} segundos para retry...`);
          await new Promise(resolve => setTimeout(resolve, retryDelay));
          
          // Aumentar o contador e o atraso para o próximo retry
          retryCount++;
          retryDelay *= 2; // Backoff exponencial
        }
      }
    }
    
    // Se chegou aqui, é porque todos os modelos falharam
    console.error('Todos os modelos Maritaca falharam');
    return null;
    
  } catch (error) {
    console.error('Erro ao consultar Maritaca:', error);
    return null;
  }
}

function findCommonResponse(claudeResponse, gptResponse, geminiResponse, deepseekResponse, maritacaResponse) {
  // Se alguma resposta estiver faltando, retorna as disponíveis
  const responses = [claudeResponse, gptResponse, geminiResponse, deepseekResponse, maritacaResponse].filter(r => r && r.trim() !== '');
  
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
    { model: 'maritaca', response: normalizeResponse(maritacaResponse), weight: MODEL_WEIGHTS['maritaca'] }
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
  
  if (claudeResponse2 && geminiResponse2 && claudeResponse2 === geminiResponse2) {
    console.log(`Consenso forte: Claude e Gemini concordam em ${claudeResponse2}`);
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

// Rota para analisar imagens
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
      const [claudeResponse, gptResponse, geminiResponse, deepseekResponse, maritacaResponse] = await Promise.all([
        askClaude(fullText, item.numero),
        askGPT(fullText, item.numero),
        askGemini(fullText, item.numero),
        askDeepSeek(fullText, item.numero),
        askMaritaca(fullText, item.numero)
      ]);

      // Encontrar resposta consensual
      const commonResponse = findCommonResponse(
        claudeResponse, 
        gptResponse, 
        geminiResponse, 
        deepseekResponse, 
        maritacaResponse
      );

      // Solicitar uma justificativa ao Claude (geralmente é o mais preciso em explicações)
      let justificativa = '';
      try {
        if (anthropic) {
          const justPrompt = `
${extractedData.texto_principal}

Item ${item.numero}: ${item.afirmacao}

Este item foi avaliado como ${commonResponse}.

Por favor, forneça uma justificativa concisa para esta resposta, explicando por que o item é ${commonResponse} com base no texto.
Mantenha a explicação objetiva e direta, com no máximo 2-3 frases.
`;

          for (const modelName of CLAUDE_MODELS) {
            try {
              const justResponse = await anthropic.messages.create({
                model: modelName,
                max_tokens: 150,
                messages: [
                  { role: 'user', content: justPrompt }
                ]
              });
              
              justificativa = justResponse.content[0].text.trim();
              break; // Se conseguiu, sai do loop
            } catch (err) {
              console.log(`Erro ao obter justificativa do modelo ${modelName}, tentando outro...`);
            }
          }
        }
      } catch (error) {
        console.error('Erro ao obter justificativa:', error);
        justificativa = 'Não foi possível gerar uma justificativa.';
      }

      if (!justificativa) {
        justificativa = 'Não foi possível gerar uma justificativa.';
      }

      return {
        ...item,
        resposta: commonResponse,
        justificativa: justificativa,
        respostas_modelos: {
          claude: claudeResponse,
          gpt: gptResponse,
          gemini: geminiResponse,
          deepseek: deepseekResponse,
          maritaca: maritacaResponse
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

// Rota básica para verificar se a API está funcionando
app.get('/', (req, res) => {
  res.json({
    status: 'online',
    message: 'API de análise de imagens com IA está funcionando'
  });
});

// Iniciar o servidor
app.listen(port, () => {
  console.log(`Servidor rodando na porta ${port}`);
});

module.exports = app;