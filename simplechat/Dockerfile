FROM node:20
WORKDIR /app
COPY package*.json ./
RUN rm -rf .next/
RUN rm -rf node_modules/
RUN npm install -g npm@10.9.0
RUN npm i sharp
RUN npm update -g
RUN npm install
COPY . .
RUN npm run build
EXPOSE 3000
CMD ["npm", "run", "start"]