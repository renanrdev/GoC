const fs = require('fs');
const path = require('path');
const { OpenAI } = require('openai');

/**
 * Extrai texto de uma imagem usando GPT-4 Vision 
 * @param {string} imagePath - Caminho da imagem
 * @param {Object} options - Opções adicionais
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

    // Prompt para extrair texto e identificar itens, adaptado para detectar questões discursivas
    const extractionPrompt = `
Analise cuidadosamente a imagem que contém um texto com questões.

As questões podem ser de dois tipos:
1. Itens de CERTO/ERRADO (verdadeiro/falso)
2. Questões DISCURSIVAS que requerem uma resposta elaborada

IMPORTANTE: 
- Mantenha EXATAMENTE os mesmos números que aparecem à esquerda de cada item/questão na imagem.
- Por exemplo, se a imagem mostrar itens 57, 58, 59 e 60, preserve estes números específicos.
- Capture o texto completo de cada item ou questão.
- Não faça nenhuma análise ou julgamento sobre os itens/questões, apenas extraia o conteúdo.

Forneça as informações em um formato JSON estruturado:

{
  "texto_principal": "transcrição do texto principal da questão",
  "itens": [
    {
      "numero": "57",  // Use EXATAMENTE o número mostrado na imagem, não renumere
      "afirmacao": "texto completo da afirmação ou pergunta"
    },
    {
      "numero": "58",  // Use EXATAMENTE o número mostrado na imagem, não renumere
      "afirmacao": "texto completo da afirmação ou pergunta"
    },
    // ... outros itens com seus números originais
  ]
}

Instruções importantes:
- Seja extremamente preciso na transcrição do texto
- Extraia TODOS os itens/questões visíveis na imagem
- Preserve os números EXATAMENTE como aparecem na imagem (ex: 57, 58, 59, 60)
- Preserve a formatação original incluindo parênteses, citações, etc.
- Se houver qualquer dúvida ou texto ilegível, indique claramente
`;

    // Chamar API de visão do GPT-4
    const response = await openai.chat.completions.create({
      model: "gpt-4o",
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
 * Detecta automaticamente se as questões são de certo/errado ou discursivas
 * @param {Object} extractedData - Dados extraídos da imagem
 * @returns {Promise<string>} - 'certo_errado' ou 'discursiva'
 */
async function detectQuestionType(extractedData) {
  try {
    // Verificar se a API key está configurada
    if (!process.env.OPENAI_API_KEY) {
      throw new Error('Chave de API da OpenAI não configurada');
    }

    // Inicializar cliente OpenAI
    const openai = new OpenAI({
      apiKey: process.env.OPENAI_API_KEY
    });
    
    // Preparar texto de exemplo para análise
    let sampleText = extractedData.texto_principal;
    
    // Adicionar até 2 itens de exemplo (se disponíveis)
    const numExamples = Math.min(2, extractedData.itens.length);
    for (let i = 0; i < numExamples; i++) {
      sampleText += `\n\nItem ${extractedData.itens[i].numero}: ${extractedData.itens[i].afirmacao}`;
    }
    
    // Criar prompt para detecção do tipo de questão
    const detectionPrompt = `
Analise cuidadosamente o seguinte texto que contém parte de uma prova ou questionário:

${sampleText}

Determine se este é um conjunto de:
1) Questões de CERTO/ERRADO (verdadeiro/falso) - onde cada item deve ser classificado como verdadeiro ou falso
2) Questões DISCURSIVAS - onde cada item é uma pergunta que requer uma resposta elaborada

Retorne APENAS uma das seguintes respostas:
- "certo_errado": se os itens são afirmações para serem classificadas como verdadeiras ou falsas
- "discursiva": se os itens são perguntas que requerem respostas elaboradas

Fatores a considerar:
- Questões certo/errado geralmente são afirmações declarativas
- Questões discursivas geralmente contêm palavras interrogativas (o que, como, por que, etc.)
- Questões certo/errado não pedem explicações, apenas julgamento da veracidade
- Questões discursivas geralmente solicitam explicações, análises ou desenvolvimentos
`;

    // Consultar o modelo para classificação
    const classification = await openai.chat.completions.create({
      model: "gpt-3.5-turbo",
      messages: [{ role: "user", content: detectionPrompt }],
      max_tokens: 50,
      temperature: 0.1
    });
    
    const response = classification.choices[0].message.content.trim().toLowerCase();
    
    // Normalizar resposta
    if (response.includes('certo_errado') || 
        response.includes('certo/errado') || 
        response.includes('verdadeiro/falso') || 
        response.includes('verdadeiro ou falso')) {
      return 'certo_errado';
    } else if (response.includes('discursiva')) {
      return 'discursiva';
    }
    
    // Se não conseguiu determinar com certeza, verificar características do texto
    // como quantidade de pontos de interrogação, palavras interrogativas, etc.
    
    let interrogativeScore = 0;
    const interrogativeWords = ['quem', 'qual', 'quais', 'que', 'como', 'por que', 'onde', 'quando', 'explique', 'descreva', 'discuta', 'analise', 'comente', 'compare'];
    
    // Verificar cada item
    for (const item of extractedData.itens) {
      const lowerText = item.afirmacao.toLowerCase();
      
      // Pontos por cada símbolo de interrogação
      const questionMarks = (lowerText.match(/\?/g) || []).length;
      interrogativeScore += questionMarks * 2;
      
      // Pontos por palavras interrogativas
      for (const word of interrogativeWords) {
        if (lowerText.includes(word)) {
          interrogativeScore++;
        }
      }
    }
    
    // Pontuar com base no número médio de palavras por item (questões discursivas tendem a ser mais longas)
    const avgWordCount = extractedData.itens.reduce((sum, item) => {
      return sum + item.afirmacao.split(/\s+/).length;
    }, 0) / extractedData.itens.length;
    
    // Se média de palavras por item for maior que 20, adicionar pontos para discursiva
    if (avgWordCount > 20) {
      interrogativeScore += 2;
    }
    
    // Decisão final
    return interrogativeScore >= 2 ? 'discursiva' : 'certo_errado';
  } catch (error) {
    console.error('Erro ao detectar tipo de questão:', error);
    // Em caso de erro, assumir que é questão de certo/errado (comportamento padrão)
    return 'certo_errado';
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

  const isDiscursivo = analysisData.tipo_questao === 'discursiva';
  
  let result = "RESULTADO DA ANÁLISE:\n\n";
  result += `Tipo de questão: ${isDiscursivo ? 'DISCURSIVA' : 'CERTO/ERRADO'}\n\n`;
  
  if (isDiscursivo) {
    analysisData.itens.forEach(item => {
      result += `Questão ${item.numero}:\n`;
      result += `${item.afirmacao}\n\n`;
      result += `Resposta recomendada:\n${item.resposta}\n\n`;
      result += `---------------------------\n\n`;
    });
  } else {
    analysisData.itens.forEach(item => {
      result += `Item ${item.numero}: ${item.resposta}\n`;
      if (item.justificativa) {
        result += `Justificativa: ${item.justificativa}\n`;
      }
      result += `\n`;
    });
  }

  return result;
}

module.exports = {
  extractTextFromImage,
  extractQuestion,
  formatAnalysisResult,
  detectQuestionType
};