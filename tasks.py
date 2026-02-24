from invoke import Context, task


@task
def serve(ctx: Context, ssl=True, http2=True):
    cmd = "uv run granian server:app --interface asgi --workers 4 --host 0.0.0.0"
    if ssl:
        cmd += " --ssl-keyfile server.key --ssl-certificate server.crt --port 8443"
    if not http2:
        cmd += " --http 1"
    ctx.run(cmd)


@task
def generate_tls_certs(ctx: Context):
    ctx.run(
        'openssl req -x509 -new -nodes -keyout ca.key -out ca.crt -subj "/CN=Fake CA" -addext "keyUsage=critical,digitalSignature,keyCertSign"'
    )
    ctx.run(
        "openssl req -new -newkey rsa:2048 -nodes -keyout server.key -out server.csr "
        '-subj "/CN=localhost" '
        '-addext "subjectAltName=DNS:localhost,IP:127.0.0.1"'
    )
    ctx.run(
        "openssl x509 -req -in server.csr -CA ca.crt -CAkey ca.key -CAcreateserial -out server.crt -days 36500 "
        '-extfile <(printf "subjectAltName=DNS:localhost,IP:127.0.0.1")'
    )
