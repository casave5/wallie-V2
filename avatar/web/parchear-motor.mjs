import fs from 'node:fs';
import path from 'node:path';

const RAIZ = path.dirname(new URL(import.meta.url).pathname);
const ORIGEN = path.join(
  RAIZ,
  'node_modules/untitled-pixi-live2d-engine/dist/cubism.min.js',
);
const DESTINO = path.join(RAIZ, 'public/vendor/cubism.min.js');

const original = 'throw new Error("Failed to upload Live2D texture.")';
const parcheado =
  'throw new Error("Failed to upload Live2D texture. ORIGEN>>name=" + (M && M.name) + " msg=" + (M && M.message) + " str=" + String(M) + " pila=" + (M && M.stack))';

fs.mkdirSync(path.dirname(DESTINO), { recursive: true });

let texto = fs.readFileSync(ORIGEN, 'utf8');
const occurrences = texto.split(original).length - 1;

if (occurrences === 0) {
  if (texto.includes('ORIGEN>>')) {
    console.log('ya estaba parcheado');
  } else {
    console.error('no se encontro la sentencia a parchear');
    process.exit(1);
  }
} else {
  texto = texto.split(original).join(parcheado);
  fs.writeFileSync(DESTINO, texto);
  console.log(`parcheado (${occurrences} sitio) -> ${DESTINO}`);
  console.log(`tamaño: ${(texto.length / 1048576).toFixed(2)} MB`);
}
