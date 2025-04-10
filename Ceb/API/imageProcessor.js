const fs = require('fs');
const path = require('path');
const { OpenAI } = require('openai');

/**
 * Extrai texto de uma imagem usando GPT-4 Vision 
 * @param {string} imagePath - Caminho da imagem
 * @param {Object} options - Opções adicionais
 * @param {Object} models - Referências às funções de consulta de modelos
 * @returns {Promise<Object>} Objeto com texto e itens extraídos
 */
async function extractTextFromImage(imagePath, options = {}) {
  try {
    // Verificar se a API key está configurada
    if (!process.env.OPENAI_API_KEY) {
      throw new Error('Chave de API da OpenAI não configurada');
    }

    // Inicializar cliente OpenAI
    const openai = new OpenAI({
      apiKey: process.env.OPENAI_API_KEY
    });

    // Ler a imagem como buffer
    const imageBuffer = fs.readFileSync(imagePath);
    const base64Image = imageBuffer.toString('base64');

    // Prompt para extrair texto e identificar itens
    const extractionPrompt = `
Analise cuidadosamente a imagem que contém um texto e itens numerados.

Extraia o texto principal e identifique cada item numerado que precisa ser avaliado como verdadeiro ou falso.
Não faça nenhuma análise ou julgamento sobre os itens, apenas extraia o conteúdo.

Forneça as informações em um formato JSON estruturado:

{
  "texto_principal": "transcrição do texto principal da questão",
  "itens": [
    {
      "numero": "1",
      "afirmacao": "texto completo da afirmação 1"
    },
    {
      "numero": "2",
      "afirmacao": "texto completo da afirmação 2"
    },
    // ... outros itens
  ]
}

Instruções importantes:
- Seja extremamente preciso na transcrição do texto
- Extraia TODOS os itens visíveis na imagem
- Preserve a formatação original incluindo parênteses, citações, etc.
- Se houver qualquer dúvida ou texto ilegível, indique claramente
`;

    // Chamar API de visão do GPT-4
    const response = await openai.chat.completions.create({
      model: "gpt-4.5-preview",
      messages: [
        {
          role: "user",
          content: [
            { type: "text", text: extractionPrompt },
            {
              type: "image_url",
              image_url: {
                url: `data:image/png;base64,${base64Image}`,
              },
            },
          ],
        },
      ],
      response_format: { type: "json_object" },
      max_tokens: 3000,
    });

    // Extrair e parsear o texto
    const extractedData = JSON.parse(response.choices[0].message.content);

    // Verificar se dados foram extraídos com sucesso
    if (!extractedData.itens || extractedData.itens.length === 0) {
      throw new Error('Não foi possível extrair informações da imagem ou identificar itens para análise');
    }

    console.log(`Texto principal extraído com sucesso. ${extractedData.itens.length} itens identificados.`);

    return extractedData;
  } catch (error) {
    console.error('Erro na extração de texto:', error);
    throw error;
  }
}

/**
 * Extrai a pergunta do objeto de análise
 * @param {Object} extractedData - Dados extraídos pelo GPT
 * @returns {string} Pergunta identificada
 */
function extractQuestion(extractedData) {
  // Se o dado já foi extraído pelo GPT, retornar o texto principal
  if (extractedData && extractedData.texto_principal) {
    return extractedData.texto_principal;
  }

  // Fallback para caso de erro
  return "Texto não identificado";
}

/**
 * Formata o resultado da análise para exibição
 * @param {Object} analysisData - Dados da análise
 * @returns {string} Texto formatado com as respostas
 */
function formatAnalysisResult(analysisData) {
  if (!analysisData || !analysisData.itens) {
    return "Não foi possível analisar as questões";
  }

  let result = "RESULTADO DA ANÁLISE:\n\n";
  
  analysisData.itens.forEach(item => {
    result += `Item ${item.numero}: ${item.resposta}\n`;
    result += `Justificativa: ${item.justificativa}\n\n`;
  });

  return result;
}

module.exports = {
  extractTextFromImage,
  extractQuestion,
  formatAnalysisResult
};