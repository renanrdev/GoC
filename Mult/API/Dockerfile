FROM node:23-slim

RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    tesseract-ocr-por \
    libpng-dev \
    libjpeg-dev \
    libgif-dev \
    librsvg2-dev \
    poppler-utils \
    imagemagick \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY package*.json ./

RUN npm install

COPY . .

RUN mkdir -p uploads responses

ENV PORT=3000
ENV NODE_ENV=production

EXPOSE 3000

CMD ["node", "app.js"]