// app.js - Servidor principal da API
const express = require('express');
const multer = require('multer');
const cors = require('cors');
const Tesseract = require('tesseract.js');
const { Anthropic } = require('@anthropic-ai/sdk');
const { OpenAI } = require('openai');
const { GoogleGenerativeAI } = require('@google/generative-ai');
const fs = require('fs');
const util = require('util');
const path = require('path');
const dotenv = require('dotenv');

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

// Configuração do Multer para upload de arquivos
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
  limits: { fileSize: 5 * 1024 * 1024 }, // Limite de 5MB
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

function withTimeout(promise, timeoutMs = 10000, errorMessage = 'Operação excedeu o tempo limite') {
  // Cria uma promise que rejeita após o tempo definido
  const timeoutPromise = new Promise((_, reject) => {
    const id = setTimeout(() => {
      clearTimeout(id);
      reject(new Error(errorMessage));
    }, timeoutMs);
  });

  // Retorna a primeira promise que resolver ou rejeitar
  return Promise.race([promise, timeoutPromise]);
}


// Função para extrair texto de uma imagem usando OCR
async function extractTextFromImage(imagePath) {
  try {
    const { data } = await Tesseract.recognize(
      imagePath,
      'por+eng', // Suporte para português e inglês
      //{ logger: m => console.log(m) }
    );
    
    return data.text.trim();
  } catch (error) {
    console.error('Erro ao extrair texto da imagem:', error);
    throw new Error('Falha ao processar a imagem com OCR');
  }
}

// Função para identificar a pergunta no texto extraído
function extractQuestion(text) {
  // Estratégia simples: procurar por frases que terminam com ponto de interrogação
  const questionRegex = /[^.!?]*\?/g;
  const questions = text.match(questionRegex);
  
  if (questions && questions.length > 0) {
    return questions[0].trim();
  }
  
  // Se não encontrar um ponto de interrogação, assume que o texto inteiro é a pergunta
  return text;
}

// Função para obter resposta do Claude com retry automático
async function askClaude(question) {
  try {
    if (!anthropic) {
      console.log('Cliente do Claude não está configurado');
      return null;
    }
    
    // Modificar o prompt para obter apenas a alternativa correta
    const enhancedPrompt = `
${question}

INSTRUÇÕES IMPORTANTES:
- Responda APENAS com a letra da alternativa correta (A, B, C, D ou E)
- NÃO forneça explicações ou justificativas
- Retorne SOMENTE a alternativa correta, ex: "A alternativa correta é (B)"
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
              max_tokens: 1000,
              messages: [
                { role: 'user', content: enhancedPrompt }
              ]
            });
            
            const responseText = response.content[0].text;
            
            // Processar a resposta para extrair apenas a alternativa
            const alternativeMatch = responseText.match(/alternativa correta [éeh\s:]+([(]?)([A-E])([)]?)/i);
            if (alternativeMatch) {
              return `A alternativa correta é (${alternativeMatch[2]})`;
            }
            
            // Se não encontrou o padrão específico, tenta outro formato
            const letterMatch = responseText.match(/^[^A-Za-z]*([A-E])[^A-Za-z]*$/);
            if (letterMatch) {
              return `A alternativa correta é (${letterMatch[1]})`;
            }
            
            // Retornar o texto original se não encontrou nenhum padrão conhecido
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

async function askGPT(question) {
    try {
      if (!openai) {
        console.log('Cliente do GPT não está configurado');
        return null;
      }
      
      // Prompt melhorado com instruções mais claras e formato estruturado
      const enhancedPrompt = `
  ${question}
  
  INSTRUÇÕES IMPORTANTÍSSIMAS (SIGA EXATAMENTE ESTE FORMATO):
  1. FORMATO OBRIGATÓRIO: "A alternativa correta é (X)" onde X é a letra A, B, C, D ou E.
  2. NÃO REPITA o texto das alternativas.
  3. NÃO inclua explicações ou justificativas.
  4. NÃO use formatação adicional.
  5. APENAS retorne a resposta no formato solicitado.
  
  EXEMPLO DE RESPOSTA DESEJADA:
  "A alternativa correta é (B)"
  
  EXEMPLO DE RESPOSTA INDESEJADA:
  "A) Somente as proposições I, II e IV estão corretas."
  
  LEMBRE-SE: Responda APENAS com "A alternativa correta é (X)" e nada mais.
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
                { role: 'system', content: 'Você é um assistente especializado em responder questões de múltipla escolha com extrema precisão e concisão. Siga EXATAMENTE o formato solicitado.' },
                { role: 'user', content: enhancedPrompt }
              ],
              max_tokens: 50,  // Reduzido para evitar respostas longas
              temperature: 0.1 // Temperatura baixa para respostas mais previsíveis
            });
            
            const responseText = response.choices[0].message.content.trim();
            
            // Processamento mais robusto para extrair a alternativa correta
            const alternativeMatch = responseText.match(/alternativa correta [éeh\s:]+([(]?)([A-E])([)]?)/i);
            if (alternativeMatch) {
              return `A alternativa correta é (${alternativeMatch[2]})`;
            }
            
            // Verificar formato mais simples - apenas letra entre parênteses
            const parenMatch = responseText.match(/\(([A-E])\)/i);
            if (parenMatch) {
              return `A alternativa correta é (${parenMatch[1]})`;
            }
            
            // Verificar se a resposta é apenas uma letra
            const letterMatch = responseText.match(/^[^A-Za-z]*([A-E])[^A-Za-z]*$/i);
            if (letterMatch) {
              return `A alternativa correta é (${letterMatch[1]})`;
            }
            
            // Verificar se a resposta contém algo como "Letra X" ou "Opção X"
            const letterRefMatch = responseText.match(/(?:letra|opção|alternativa)\s+([A-E])/i);
            if (letterRefMatch) {
              return `A alternativa correta é (${letterRefMatch[1]})`;
            }
            
            // Se ainda não conseguiu extrair um padrão conhecido, mas tem uma letra válida em algum lugar
            const anyLetterMatch = responseText.match(/([A-E])[^A-Za-z]/i);
            if (anyLetterMatch) {
              return `A alternativa correta é (${anyLetterMatch[1]})`;
            }
            
            // Se ainda não conseguimos extrair, tentar novamente com outro modelo
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
async function askGemini(question) {
  try {
    if (!genAI) {
      console.log('Cliente do Gemini não está configurado');
      return null;
    }
    
    // Modificar o prompt para obter apenas a alternativa correta
    const enhancedPrompt = `
${question}

INSTRUÇÕES IMPORTANTES:
- Responda APENAS com a letra da alternativa correta (A, B, C, D ou E)
- NÃO forneça explicações ou justificativas
- Retorne SOMENTE a alternativa correta, ex: "A alternativa correta é (B)"
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
          const text = response.text();
          
          // Processar a resposta para extrair apenas a alternativa
          const alternativeMatch = text.match(/alternativa correta [éeh\s:]+([(]?)([A-E])([)]?)/i);
          if (alternativeMatch) {
            return `A alternativa correta é (${alternativeMatch[2]})`;
          }
          
          // Se não encontrou o padrão específico, tenta outro formato
          const letterMatch = text.match(/^[^A-Za-z]*([A-E])[^A-Za-z]*$/);
          if (letterMatch) {
            return `A alternativa correta é (${letterMatch[1]})`;
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
async function askDeepSeek(question) {
  try {
    if (!deepseek) {
      console.log('Cliente do DeepSeek não está configurado');
      return null;
    }
    
    // Modificar o prompt para obter apenas a alternativa correta
    const enhancedPrompt = `
${question}

INSTRUÇÕES IMPORTANTES:
- Responda APENAS com a letra da alternativa correta (A, B, C, D ou E)
- NÃO forneça explicações ou justificativas
- Retorne SOMENTE a alternativa correta, ex: "A alternativa correta é (B)"
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
            max_tokens: 1000
          });
          
          const responseText = response.choices[0].message.content;
          
          // Processar a resposta para extrair apenas a alternativa
          const alternativeMatch = responseText.match(/alternativa correta [éeh\s:]+([(]?)([A-E])([)]?)/i);
          if (alternativeMatch) {
            return `A alternativa correta é (${alternativeMatch[2]})`;
          }
          
          // Se não encontrou o padrão específico, tenta outro formato
          const letterMatch = responseText.match(/^[^A-Za-z]*([A-E])[^A-Za-z]*$/);
          if (letterMatch) {
            return `A alternativa correta é (${letterMatch[1]})`;
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
async function askMaritaca(question) {
  try {
    if (!maritaca) {
      console.log('Cliente do Maritaca não está configurado');
      return null;
    }
    
    // Modificar o prompt para obter apenas a alternativa correta
    const enhancedPrompt = `
${question}

INSTRUÇÕES IMPORTANTES:
- Responda APENAS com a letra da alternativa correta (A, B, C, D ou E)
- NÃO forneça explicações ou justificativas
- Retorne SOMENTE a alternativa correta, ex: "A alternativa correta é (B)"
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
            max_tokens: 1000
          });
          
          const responseText = response.choices[0].message.content;
          
          // Processar a resposta para extrair apenas a alternativa
          const alternativeMatch = responseText.match(/alternativa correta [éeh\s:]+([(]?)([A-E])([)]?)/i);
          if (alternativeMatch) {
            return `A alternativa correta é (${alternativeMatch[2]})`;
          }
          
          // Se não encontrou o padrão específico, tenta outro formato
          const letterMatch = responseText.match(/^[^A-Za-z]*([A-E])[^A-Za-z]*$/);
          if (letterMatch) {
            return `A alternativa correta é (${letterMatch[1]})`;
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
      console.log('Apenas uma resposta válida disponível:', responses[0].substring(0, 50) + '...');
      return responses[0];
    }
    
    // Extrair apenas as letras das alternativas para comparação
    const extractAlternative = (text) => {
      if (!text) return null;
      
      // Padrão para capturar alternativa no formato "A alternativa correta é (X)"
      const match1 = text.match(/alternativa correta [éeh\s:]+([(]?)([A-E])([)]?)/i);
      if (match1) return match1[2].toUpperCase();
      
      // Padrão para capturar apenas a letra
      const match2 = text.match(/^[^A-Za-z]*([A-E])[^A-Za-z]*$/);
      if (match2) return match2[1].toUpperCase();
      
      // Padrão para encontrar uma letra entre parênteses
      const match3 = text.match(/\(([A-E])\)/);
      if (match3) return match3[1].toUpperCase();
      
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
    
    // Extrair as alternativas com seu modelo de origem
    const allResponsesWithAlternatives = [
      { model: 'claude', response: claudeResponse, alternative: claudeResponse ? extractAlternative(claudeResponse) : null, weight: MODEL_WEIGHTS['claude'] },
      { model: 'gpt', response: gptResponse, alternative: gptResponse ? extractAlternative(gptResponse) : null, weight: MODEL_WEIGHTS['gpt'] },
      { model: 'gemini', response: geminiResponse, alternative: geminiResponse ? extractAlternative(geminiResponse) : null, weight: MODEL_WEIGHTS['gemini'] },
      { model: 'deepseek', response: deepseekResponse, alternative: deepseekResponse ? extractAlternative(deepseekResponse) : null, weight: MODEL_WEIGHTS['deepseek'] },
      { model: 'maritaca', response: maritacaResponse, alternative: maritacaResponse ? extractAlternative(maritacaResponse) : null, weight: MODEL_WEIGHTS['maritaca'] }
    ].filter(item => item.alternative !== null);
    
    const alternatives = allResponsesWithAlternatives.map(item => item.alternative);
    
    console.log('Alternativas extraídas:', alternatives.join(', '));
    
    // Contar ocorrências
    const counts = {};
    const modelsByAlternative = {};
    const weightSumByAlternative = {};
    
    allResponsesWithAlternatives.forEach(item => {
      const alt = item.alternative;
      counts[alt] = (counts[alt] || 0) + 1;
      
      // Registrar quais modelos escolheram esta alternativa
      if (!modelsByAlternative[alt]) {
        modelsByAlternative[alt] = [];
        weightSumByAlternative[alt] = 0;
      }
      modelsByAlternative[alt].push(item.model);
      weightSumByAlternative[alt] += item.weight;
    });
    
    // Primeiro: verificar se há alternativa com 3+ votos
    let maxCount = 0;
    let alternatives3PlusVotes = [];
    
    for (const alt in counts) {
      if (counts[alt] > maxCount) {
        maxCount = counts[alt];
      }
      
      if (counts[alt] >= 3) {
        alternatives3PlusVotes.push(alt);
      }
    }
    
    if (alternatives3PlusVotes.length === 1) {
      const winner = alternatives3PlusVotes[0];
      console.log(`Consenso forte: alternativa ${winner} com ${counts[winner]} votos de ${modelsByAlternative[winner].join(', ')}`);
      return `A alternativa correta é (${winner})`;
    }
    
    // Se tiver mais de uma alternativa com 3+ votos (raro, mas possível)
    if (alternatives3PlusVotes.length > 1) {
      // Desempatar por peso dos modelos
      let bestAlternative = null;
      let highestWeight = 0;
      
      alternatives3PlusVotes.forEach(alt => {
        if (weightSumByAlternative[alt] > highestWeight) {
          highestWeight = weightSumByAlternative[alt];
          bestAlternative = alt;
        }
      });
      
      console.log(`Múltiplas alternativas com 3+ votos. Desempate por peso: ${bestAlternative} (peso ${highestWeight})`);
      return `A alternativa correta é (${bestAlternative})`;
    }
    
    // Segundo: verificar alternativas com 2 votos
    const alternativesWith2Votes = [];
    for (const alt in counts) {
      if (counts[alt] === 2) {
        alternativesWith2Votes.push(alt);
      }
    }
    
    // Se houver apenas uma alternativa com 2 votos, verificar se inclui algum dos modelos principais
    if (alternativesWith2Votes.length === 1) {
      const alt = alternativesWith2Votes[0];
      const models = modelsByAlternative[alt];
      const hasBigThreeModel = models.some(model => ['claude', 'gemini', 'gpt'].includes(model));
      
      if (hasBigThreeModel) {
        console.log(`Consenso parcial: alternativa ${alt} com 2 votos de ${models.join(', ')}, incluindo modelo principal`);
        return `A alternativa correta é (${alt})`;
      }
    }
    
    // Se houver múltiplas alternativas com 2 votos, priorizar aquela com modelos mais confiáveis
    if (alternativesWith2Votes.length > 1) {
      // Verificar se alguma das alternativas com 2 votos tem Claude E Gemini concordando
      const claudeGeminiConsensus = alternativesWith2Votes.find(alt => {
        const models = modelsByAlternative[alt];
        return models.includes('claude') && models.includes('gemini');
      });
      
      if (claudeGeminiConsensus) {
        console.log(`Forte consenso parcial: Claude e Gemini concordam na alternativa ${claudeGeminiConsensus}`);
        return `A alternativa correta é (${claudeGeminiConsensus})`;
      }
      
      // Verificar se alguma das alternativas com 2 votos tem Claude E GPT concordando
      const claudeGptConsensus = alternativesWith2Votes.find(alt => {
        const models = modelsByAlternative[alt];
        return models.includes('claude') && models.includes('gpt');
      });
      
      if (claudeGptConsensus) {
        console.log(`Forte consenso parcial: Claude e GPT concordam na alternativa ${claudeGptConsensus}`);
        return `A alternativa correta é (${claudeGptConsensus})`;
      }
      
      // Verificar se alguma das alternativas com 2 votos tem Gemini E GPT concordando
      const geminiGptConsensus = alternativesWith2Votes.find(alt => {
        const models = modelsByAlternative[alt];
        return models.includes('gemini') && models.includes('gpt');
      });
      
      if (geminiGptConsensus) {
        console.log(`Forte consenso parcial: Gemini e GPT concordam na alternativa ${geminiGptConsensus}`);
        return `A alternativa correta é (${geminiGptConsensus})`;
      }
      
      // Se chegou aqui, escolher a alternativa com 2 votos que tem maior peso total
      let bestAlternative = null;
      let highestWeight = 0;
      
      alternativesWith2Votes.forEach(alt => {
        if (weightSumByAlternative[alt] > highestWeight) {
          highestWeight = weightSumByAlternative[alt];
          bestAlternative = alt;
        }
      });
      
      console.log(`Múltiplas alternativas com 2 votos. Desempate por peso: ${bestAlternative} (peso ${highestWeight})`);
      return `A alternativa correta é (${bestAlternative})`;
    }
    
    // Terceiro: Se chegou aqui, priorizar a resposta do modelo mais confiável
    // Prioridade: Claude > Gemini > GPT > DeepSeek > Maritaca
    const modelPriority = ['claude', 'gemini', 'gpt', 'deepseek', 'maritaca'];
    
    for (const model of modelPriority) {
      const modelResponse = allResponsesWithAlternatives.find(item => item.model === model);
      if (modelResponse) {
        console.log(`Sem consenso claro. Usando resposta do modelo prioritário: ${model} -> ${modelResponse.alternative}`);
        return `A alternativa correta é (${modelResponse.alternative})`;
      }
    }
    
    // Se chegou aqui, usar a primeira resposta disponível
    console.log('Sem nenhum consenso ou modelo prioritário disponível. Usando primeira resposta.');
    return allResponsesWithAlternatives[0].response;
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

// Endpoint principal da API
app.post('/api/analyze', upload.single('image'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'Nenhuma imagem enviada' });
    }
    
    // Extrair texto da imagem
    const imagePath = req.file.path;
    const extractedText = await extractTextFromImage(imagePath);
    
    if (!extractedText) {
      return res.status(400).json({ error: 'Não foi possível extrair texto da imagem' });
    }
    
    // Identificar a pergunta no texto
    const question = extractQuestion(extractedText);
    
    if (!question) {
      return res.status(400).json({ error: 'Nenhuma pergunta identificada no texto' });
    }
    
    // Obter respostas dos modelos de IA em paralelo
    const [claudeResponse, gptResponse, geminiResponse, deepseekResponse, maritacaResponse] = await Promise.all([
      askClaude(question),
      askGPT(question),
      askGemini(question),
      askDeepSeek(question),
      askMaritaca(question)
    ]);
    
    // Log das respostas para fins de depuração
    console.log('Claude:', claudeResponse);
    console.log('GPT:', gptResponse);
    console.log('Gemini:', geminiResponse);
    console.log('DeepSeek:', deepseekResponse);
    console.log('Maritaca:', maritacaResponse);
    
    // Verificar se pelo menos uma resposta foi obtida
    const availableResponses = [claudeResponse, gptResponse, geminiResponse, deepseekResponse, maritacaResponse].filter(r => r);
    
    if (availableResponses.length === 0) {
      return res.status(500).json({ 
        error: 'Nenhum modelo de IA disponível respondeu à pergunta',
        extractedText,
        question
      });
    }
    
    // Encontrar a resposta comum ou a melhor resposta disponível
    const commonResponse = findCommonResponse(claudeResponse, gptResponse, geminiResponse, deepseekResponse, maritacaResponse);
    
    // Salvar a resposta em um arquivo de texto
    let responseFilePath = null;
    try {
      responseFilePath = await saveResponseToFile(commonResponse);
    } catch (error) {
      console.error("Erro ao salvar resposta:", error);
    }
    
    // Construir URL para o arquivo (se existir)
    const responseUrl = responseFilePath 
      ? `${req.protocol}://${req.get('host')}/${responseFilePath}` 
      : null;
    
    res.json({
      success: true,
      extractedText,
      question,
      responses: {
        claude: claudeResponse,
        gpt: gptResponse,
        gemini: geminiResponse,
        deepseek: deepseekResponse,
        maritaca: maritacaResponse
      },
      commonResponse,
      responseUrl
    });
    
  } catch (error) {
    console.error('Erro no processamento:', error);
    res.status(500).json({ 
      error: error.message || 'Erro interno no servidor',
      details: process.env.NODE_ENV === 'development' ? error.stack : undefined
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