# Swap ONLY the two GRIMOIRE binaries into the working b70-native image.
# Everything else -- bridges, oneAPI, the recipe wrapper -- is untouched, and
# grimoire:b70-native stays on disk as the rollback.
FROM grimoire:b70-native
COPY grimoire        /usr/local/bin/grimoire.real
COPY grimoire-server /usr/local/bin/grimoire-server.real
RUN chmod +x /usr/local/bin/grimoire.real /usr/local/bin/grimoire-server.real
