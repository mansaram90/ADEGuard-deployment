FROM nginx:alpine

# Copy only the browser assets into Nginx's default static-file directory.
COPY index.html styles.css app.js /usr/share/nginx/html/

EXPOSE 80
