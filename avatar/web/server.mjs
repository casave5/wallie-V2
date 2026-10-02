import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { WebSocketServer } from 'ws';

const RAIZ = path.dirname(fileURLToPath(import.meta.url));
const PUBLICO = path.join(RAIZ, 'public');
const MODELOS = process.env.MODELOS_DIR || '/home/USUARIO/vtuber/modelos';
const PUERTO = Number(process.env.PORT || 8100);

const LIBRERIAS = {
  '/pixi.min.js': path.join(RAIZ, 'node_modules/pixi.js/dist/browser/pixi.min.js'),
  '/cubism.min.js': path.join(RAIZ, 'node_modules/pixi-live2d-display/dist/cubism4.min.js'),
};

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.png': 'image/png',
  '.moc3': 'application/octet-stream',
  '.cdi3.json': 'application/json; charset=utf-8',
  '.physics3.json': 'application/json; charset=utf-8',
  '.motion3.json': 'application/json; charset=utf-8',
};

const clientes = new Set();
let ultimoEstado = { tipo: 'params', params: {} };

function tipoDe(ruta) {
  if (ruta.endsWith('.model3.json')) return 'application/json; charset=utf-8';
  return MIME[path.extname(ruta)] || 'application/octet-stream';
}

function resolver(base, relativa) {
  const limpio = path.posix.normalize('/' + relativa).replace(/^\/+/, '');
  const destino = path.resolve(base, limpio);
  if (destino !== path.resolve(base) && !destino.startsWith(path.resolve(base) + path.sep)) {
    return null;
  }
  return destino;
}

function enviarArchivo(res, ruta, codigo = 200) {
  fs.readFile(ruta, (err, datos) => {
    if (err) {
      res.writeHead(404, { 'Content-Type': 'text/plain' });
      res.end('no encontrado');
      return;
    }
    res.writeHead(codigo, {
      'Content-Type': tipoDe(ruta),
      'Cache-Control': 'no-cache',
      'Access-Control-Allow-Origin': '*',
    });
    res.end(datos);
  });
}

function difundir(mensaje) {
  const texto = JSON.stringify(mensaje);
  for (const ws of clientes) {
    if (ws.readyState === 1) ws.send(texto);
  }
}

function leerCuerpo(req) {
  return new Promise((resolver) => {
    let datos = '';
    req.on('data', (trozo) => {
      datos += trozo;
      if (datos.length > 1_000_000) req.destroy();
    });
    req.on('end', () => {
      try {
        resolver(datos ? JSON.parse(datos) : {});
      } catch {
        resolver({});
      }
    });
    req.on('error', () => resolver({}));
  });
}

const servidor = http.createServer(async (req, res) => {
  const url = new URL(req.url, `http://${req.headers.host || 'localhost'}`);
  const ruta = decodeURIComponent(url.pathname);
  if (process.env.VERBOSO === '1') {
    console.log(`${req.method} ${ruta}`);
  }

  if (req.method === 'OPTIONS') {
    res.writeHead(204, {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET,POST,OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type',
    });
    res.end();
    return;
  }

  if (ruta === '/api/retraso') {
    const ms = Math.min(Number(url.searchParams.get('ms') || 0), 30000);
    const png = Buffer.from(
      'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==',
      'base64',
    );
    setTimeout(() => {
      res.writeHead(200, { 'Content-Type': 'image/png', 'Content-Length': png.length });
      res.end(png);
    }, ms);
    return;
  }

  if (ruta === '/api/log' && req.method === 'POST') {
    const cuerpo = await leerCuerpo(req);
    const etiqueta = cuerpo.etiqueta || 'pagina';
    const texto = cuerpo.texto ?? cuerpo.mensaje ?? JSON.stringify(cuerpo);
    console.log(`[${etiqueta}] ${texto}`);
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ ok: true }));
    return;
  }

  if (ruta === '/api/estado') {
    res.writeHead(200, { 'Content-Type': 'application/json; charset=utf-8' });
    res.end(JSON.stringify({ ...ultimoEstado, clientes: clientes.size }));
    return;
  }

  if (ruta === '/api/params' && req.method === 'POST') {
    const cuerpo = await leerCuerpo(req);
    const params = {};
    for (const [clave, valor] of Object.entries(cuerpo.params || cuerpo)) {
      const numero = Number(valor);
      if (Number.isFinite(numero)) params[clave] = numero;
    }
    ultimoEstado = { tipo: 'params', params: { ...ultimoEstado.params, ...params } };
    difundir(ultimoEstado);
    res.writeHead(200, { 'Content-Type': 'application/json; charset=utf-8' });
    res.end(JSON.stringify({ ok: true, recibidos: Object.keys(params).length, clientes: clientes.size }));
    return;
  }

  // Gesto: le dice al/los visor(es) que repliquen una animacion (ej: saludo).
  if (ruta === '/api/gesto' && req.method === 'POST') {
    const cuerpo = await leerCuerpo(req);
    const gesto = String(cuerpo.gesto || cuerpo.nombre || '').trim();
    if (!gesto) {
      res.writeHead(400, { 'Content-Type': 'application/json; charset=utf-8' });
      res.end(JSON.stringify({ ok: false, error: 'falta gesto' }));
      return;
    }
    const mensaje = { tipo: 'gesto', gesto };
    difundir(mensaje);
    res.writeHead(200, { 'Content-Type': 'application/json; charset=utf-8' });
    res.end(JSON.stringify({ ok: true, gesto, clientes: clientes.size }));
    return;
  }

  if (ruta === '/api/reset' && req.method === 'POST') {
    ultimoEstado = { tipo: 'params', params: {} };
    difundir(ultimoEstado);
    res.writeHead(200, { 'Content-Type': 'application/json; charset=utf-8' });
    res.end(JSON.stringify({ ok: true }));
    return;
  }

  if (LIBRERIAS[ruta]) {
    enviarArchivo(res, LIBRERIAS[ruta]);
    return;
  }

  if (ruta.startsWith('/model/')) {
    const destino = resolver(MODELOS, ruta.slice('/model/'.length));
    if (!destino) {
      res.writeHead(403);
      res.end('prohibido');
      return;
    }
    enviarArchivo(res, destino);
    return;
  }

  if (ruta === '/modelos') {
    const lista = fs
      .readdirSync(MODELOS, { withFileTypes: true })
      .filter((e) => e.isDirectory())
      .filter((e) => fs.existsSync(path.join(MODELOS, e.name, 'runtime')))
      .map((e) => e.name);
    res.writeHead(200, { 'Content-Type': 'application/json; charset=utf-8' });
    res.end(JSON.stringify(lista));
    return;
  }

  const destino = resolver(PUBLICO, ruta === '/' ? '/index.html' : ruta);
  if (!destino) {
    res.writeHead(403);
    res.end('prohibido');
    return;
  }
  enviarArchivo(res, destino);
});

const wss = new WebSocketServer({ server: servidor, path: '/ws' });

wss.on('connection', (ws) => {
  clientes.add(ws);
  ws.send(JSON.stringify({ ...ultimoEstado, clientes: clientes.size }));
  ws.on('message', (datos) => {
    let mensaje;
    try {
      mensaje = JSON.parse(datos.toString());
    } catch {
      return;
    }
    if (mensaje.tipo === 'params' && mensaje.params && typeof mensaje.params === 'object') {
      const limpio = {};
      for (const [clave, valor] of Object.entries(mensaje.params)) {
        const numero = Number(valor);
        if (Number.isFinite(numero)) limpio[clave] = numero;
      }
      ultimoEstado = { tipo: 'params', params: { ...ultimoEstado.params, ...limpio } };
      difundir(ultimoEstado);
    }
  });
  ws.on('close', () => clientes.delete(ws));
  ws.on('error', () => clientes.delete(ws));
});

servidor.listen(PUERTO, '127.0.0.1', () => {
  console.log(`Avatar Live2D en http://127.0.0.1:${PUERTO}/  (modelos: ${MODELOS})`);
});
