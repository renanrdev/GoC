// imageProcessor.js
const fs = require('fs');
const path = require('path');
const { OpenAI } = require('openai');

/**
 * Extrai e analisa texto de uma imagem usando GPT-4 Vision
 * @param {string} imagePath - Caminho da imagem
 * @returns {Promise<Object>} Objeto com detalhes da questão
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

    // Prompt detalhado para análise completa da questão
    const analysisPrompt = `
Analise cuidadosamente a imagem e forneça as seguintes informações em um formato JSON estruturado:

{
  "numero_questao": "número da questão (se disponível)",
  "enunciado": "texto completo do enunciado da questão",
  "alternativas": [
    {
      "letra": "a",
      "texto": "texto completo da alternativa A"
    },
    {
      "letra": "b",
      "texto": "texto completo da alternativa B"
    },
    // ... outras alternativas
  ],
  "tipo_exceto": true ou false (se a questão pede para identificar a alternativa EXCETO)
}

Instruções importantes:
- Seja extremamente preciso na transcrição do texto
- Mantenha a formatação original
- Se houver qualquer dúvida ou texto ilegível, indique claramente
- Preste atenção especial a detalhes como palavras com acentuação
- Extraia TODOS os detalhes visíveis na imagem
- Tenha certeza que extraiu todas as alternativas, a maioria das vezes são 5 alternativas
- Se a imagem contiver uma tabela, extraia os dados da tabela e formate-os corretamente
`;

    // Chamar API de visão do GPT-4
    const response = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        {
          role: "user",
          content: [
            { type: "text", text: analysisPrompt },
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
      max_tokens: 1000,
    });

    // Extrair e parsear o texto
    const extractedData = JSON.parse(response.choices[0].message.content);

    // Verificar se dados foram extraídos com sucesso
    if (!extractedData.enunciado) {
      throw new Error('Não foi possível extrair informações da imagem');
    }

    return extractedData;
  } catch (error) {
    console.error('Erro na análise de texto com GPT-4 Vision:', error);
    throw error;
  }
}

/**
 * Extrai a pergunta do objeto de análise
 * @param {Object} extractedData - Dados extraídos pelo GPT
 * @returns {string} Pergunta identificada
 */
function extractQuestion(extractedData) {
  // Se o dado já foi extraído pelo GPT, retornar o enunciado
  if (extractedData && extractedData.enunciado) {
    return extractedData.enunciado;
  }

  // Fallback para caso de erro
  return "Pergunta não identificada";
}

module.exports = {
  extractTextFromImage,
  extractQuestion
};