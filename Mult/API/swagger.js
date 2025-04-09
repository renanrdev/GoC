const swaggerJsDoc = require('swagger-jsdoc');

// Definição de tags para organizar a documentação
const tags = [
  {
    name: 'Status',
    description: 'Endpoints para verificar o status do serviço'
  },
  {
    name: 'Análise',
    description: 'Endpoints para análise de questões'
  }
];

// Configuração dos modelos usados na API
const modelsList = {
  claude: 'Modelos Claude da Anthropic (claude-3-7-sonnet-20250219, claude-3-5-sonnet-20241022, etc.)',
  gpt: 'Modelos GPT da OpenAI (gpt-3.5-turbo, gpt-3.5-turbo-instruct, etc.)',
  gemini: 'Modelos Gemini do Google (gemini-2.0-flash, gemini-1.0-pro, etc.)',
  deepseek: 'Modelos DeepSeek (deepseek-chat, deepseek-coder)',
  maritaca: 'Modelos Maritaca.AI (sabia-3, sabia-2)'
};

// Opções avançadas do Swagger
const swaggerOptions = {
  definition: {
    openapi: '3.0.0',
    info: {
      title: 'API de Análise de Questões com IA',
      version: '1.0.0',
      description: `
# API para análise de questões de múltipla escolha

Esta API oferece um serviço de análise de questões de múltipla escolha a partir de imagens.
O sistema utiliza OCR para extrair o texto e consulta múltiplos modelos de IA para determinar a resposta correta.

## Modelos de IA suportados

- **Claude**: ${modelsList.claude}
- **GPT**: ${modelsList.gpt}
- **Gemini**: ${modelsList.gemini}
- **DeepSeek**: ${modelsList.deepseek}
- **Maritaca**: ${modelsList.maritaca}

## Como utilizar

1. Envie a imagem da questão através do endpoint POST /api/analyze
2. A API retornará a questão extraída e a resposta consensual entre os modelos

## Configuração

A API requer as seguintes chaves de API configuradas via variáveis de ambiente:
- ANTHROPIC_API_KEY
- OPENAI_API_KEY
- GEMINI_API_KEY
- DEEPSEEK_API_KEY
- MARITACA_API_KEY
      `,
      contact: {
        name: 'Suporte',
        email: 'suporte@exemplo.com',
        url: 'https://exemplo.com/suporte'
      },
      license: {
        name: 'MIT',
        url: 'https://opensource.org/licenses/MIT',
      },
    },
    servers: [
      {
        url: 'http://localhost:3000',
        description: 'Servidor de Desenvolvimento Local'
      },
      {
        url: 'https://goc-mult.r74hlz.easypanel.host',
        description: 'Servidor de Produção'
      }
    ],
    tags: tags,
    components: {
      schemas: {
        QuestionAlternative: {
          type: 'object',
          properties: {
            letra: {
              type: 'string',
              description: 'Letra da alternativa (A, B, C, D ou E)',
              example: 'A'
            },
            texto: {
              type: 'string',
              description: 'Texto da alternativa',
              example: 'O consumo de produtos orgânicos está aumentando globalmente.'
            }
          }
        },
        AnalyzeResponse: {
          type: 'object',
          properties: {
            success: {
              type: 'boolean',
              description: 'Indica se a requisição foi processada com sucesso',
              example: true
            },
            questionNumber: {
              type: 'string',
              description: 'Número da questão extraído da imagem',
              example: '42'
            },
            fullQuestion: {
              type: 'string',
              description: 'Texto completo do enunciado da questão',
              example: 'Sobre as tendências de mercado no setor alimentício, assinale a alternativa correta:'
            },
            alternatives: {
              type: 'array',
              description: 'Lista das alternativas extraídas da questão',
              items: {
                $ref: '#/components/schemas/QuestionAlternative'
              }
            },
            isExcetoQuestion: {
              type: 'boolean',
              description: 'Indica se a questão é do tipo "EXCETO"',
              example: false
            },
            commonResponse: {
              type: 'string',
              description: 'Resposta consensual entre os modelos de IA',
              example: 'A alternativa correta é (B)'
            },
            responseUrl: {
              type: 'string',
              description: 'URL para o arquivo com a resposta salva',
              example: 'responses/response-1712345678901.txt'
            },
            responses: {
              type: 'object',
              description: 'Respostas individuais de cada modelo de IA',
              properties: {
                claude: {
                  type: 'string',
                  description: 'Resposta do modelo Claude da Anthropic',
                  example: 'A alternativa correta é (B)'
                },
                gpt: {
                  type: 'string',
                  description: 'Resposta do modelo GPT da OpenAI',
                  example: 'A alternativa correta é (B)'
                },
                gemini: {
                  type: 'string',
                  description: 'Resposta do modelo Gemini do Google',
                  example: 'A alternativa correta é (B)'
                },
                deepseek: {
                  type: 'string',
                  description: 'Resposta do modelo DeepSeek',
                  example: 'A alternativa correta é (B)'
                },
                maritaca: {
                  type: 'string',
                  description: 'Resposta do modelo Maritaca (Sabiá)',
                  example: 'A alternativa correta é (B)'
                }
              }
            }
          }
        },
        StatusResponse: {
          type: 'object',
          properties: {
            status: {
              type: 'string',
              description: 'Status atual da API',
              example: 'online'
            },
            message: {
              type: 'string',
              description: 'Mensagem descritiva',
              example: 'API de análise de imagens com IA está funcionando'
            },
            version: {
              type: 'string',
              description: 'Versão atual da API',
              example: '1.0.0'
            },
            uptime: {
              type: 'number',
              description: 'Tempo de execução do servidor em segundos',
              example: 3600
            }
          }
        },
        ErrorResponse: {
          type: 'object',
          properties: {
            error: {
              type: 'string',
              description: 'Mensagem de erro',
              example: 'Erro interno no servidor'
            },
            code: {
              type: 'string',
              description: 'Código de erro específico (quando disponível)',
              example: 'IMAGE_PROCESSING_ERROR'
            },
            details: {
              type: 'object',
              description: 'Detalhes adicionais sobre o erro (quando disponíveis)'
            }
          }
        }
      },
      securitySchemes: {
        apiKeyAuth: {
          type: 'apiKey',
          in: 'header',
          name: 'X-API-KEY',
          description: 'Chave de API para autenticação (implementação futura)'
        }
      }
    }
  },
  apis: ['./app.js'], // ou o caminho para os arquivos que contêm suas rotas documentadas
};

// Gerar a especificação Swagger
const swaggerSpec = swaggerJsDoc(swaggerOptions);

module.exports = swaggerSpec;