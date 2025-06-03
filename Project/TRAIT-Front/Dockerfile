# Dockerfile for serving React frontend in development
FROM node:18-slim

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci

# Copy source code
COPY . .

# Expose port for development server (matching your Vite config)
EXPOSE 5173

# Start the React development server on port 5173 to match your setup
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "5173"]