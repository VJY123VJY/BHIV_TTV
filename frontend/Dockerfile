# ==============================================================================
# BHIV TTV Frontend Production Dockerfile
# ==============================================================================
FROM nginx:alpine

# Install curl for container healthcheck
RUN apk add --no-cache curl

# Reconfigure default Nginx server to listen on port 8000 instead of 80
RUN sed -i 's/listen       80;/listen       8000;/g' /etc/nginx/conf.d/default.conf && \
    sed -i 's/listen  \[::\]:80;/listen  \[::\]:8000;/g' /etc/nginx/conf.d/default.conf

# Copy static frontend assets (index.html, app.js, style.css)
COPY . /usr/share/nginx/html/

# Create symlink so that requests to /static/* (e.g. /static/style.css, /static/app.js)
# resolve directly to the static assets root
RUN ln -s /usr/share/nginx/html /usr/share/nginx/html/static && \
    rm -f /usr/share/nginx/html/Dockerfile

# Expose microservice port
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/ || exit 1

CMD ["nginx", "-g", "daemon off;"]
