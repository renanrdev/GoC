const path = require('path');

module.exports = {
  // Configurações padrão para OCR
  tesseract: {
    // Linguagens padrão para reconhecimento
    defaultLanguages: 'por+eng',
    
    // Caminho padrão para arquivos de linguagem do Tesseract
    defaultTessDataPath: path.join(__dirname, 'tessdata'),
    
    // Opções de configuração avançadas
    config: {
      // Lista de caracteres permitidos
      tessedit_char_whitelist: 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,()[]{}:;-_/\\%',
      
      // Preservar certeza entre palavras
      preserve_interword_certainty: true,
      
      // Tamanho mínimo de linha
      textord_min_linesize: 1.5,
      
      // Inverter cores para melhorar leitura
      tessedit_do_invert: true,
    },
    
    // Modos de segmentação de página para tentar
    psmModes: [
      6,  // Bloco de texto uniforme
      11, // Texto sem layout definido
      4,  // Múltiplas colunas
      3   // Múltiplas linhas
    ]
  },
  
  // Configurações de pré-processamento de imagem
  imagePreprocessing: {
    // Opções padrão para Sharp
    sharpOptions: {
      grayscale: true,
      normalize: true,
      sharpen: true,
      threshold: true
    },
    
    // Tamanho máximo de imagem para processamento
    maxImageSizeMB: 10
  }
};