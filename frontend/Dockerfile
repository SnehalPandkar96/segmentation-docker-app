# Use Node.js official image
FROM node:18-alpine

# Set working directory
WORKDIR /app

# Copy only package.json and lock file first for caching
COPY package.json package-lock.json ./

# Install dependencies
RUN npm install

# Copy rest of the frontend code
COPY . .

# Expose the React dev server port
EXPOSE 3000

# Start the React dev server
CMD ["npm", "start"]
