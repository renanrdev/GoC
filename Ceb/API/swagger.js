const swaggerJsDoc = require('swagger-jsdoc');

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
      
    },
    servers: [
      {
        url: 'http://localhost:3000',
        description: 'Servidor de Desenvolvimento Local'
      },
      {
        url: 'https://goc-ceb.r74hlz.easypanel.host',
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
  apis: ['./app.js'], 
};

const swaggerSpec = swaggerJsDoc(swaggerOptions);

module.exports = swaggerSpec;