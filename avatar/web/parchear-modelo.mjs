import fs from 'node:fs';
import path from 'node:path';

const RAIZ_MODELOS = process.env.MODELOS_DIR || '/home/USUARIO/vtuber/modelos';

function parametrosDe(cdi) {
  const p = (cdi && cdi.Parameters) || [];
  return p.map((x) => x.Id);
}

function elegir(ids, expresiones, respaldo) {
  for (const re of expresiones) {
    const encontrados = ids.filter((id) => re.test(id));
    if (encontrados.length) return encontrados;
  }
  return respaldo ? ids.filter((id) => respaldo.test(id)) : [];
}

function parchear(nombre) {
  const runtime = path.join(RAIZ_MODELOS, nombre, 'runtime');
  const original = path.join(runtime, `${nombre}.model3.json`);
  if (!fs.existsSync(original)) {
    console.log(`${nombre}: no encontrado`);
    return;
  }
  const json = JSON.parse(fs.readFileSync(original, 'utf8'));
  const cdi = JSON.parse(fs.readFileSync(path.join(runtime, `${nombre}.cdi3.json`), 'utf8'));
  const ids = parametrosDe(cdi);

  const ojos = elegir(ids, [/^PARAM_EYE_[LR]_OPEN$/i, /EYE.*OPEN/i, /EyeBlink/i], /EYE.*OPEN/i);
  const boca = elegir(ids, [/^PARAM_MOUTH_OPEN_Y$/i, /MOUTH.*OPEN/i, /LIP.*(OPEN|Y)$/i], /MOUTH/i);

  json.Groups = [
    { Target: 'Parameter', Name: 'EyeBlink', Ids: ojos },
    { Target: 'Parameter', Name: 'LipSync', Ids: boca.slice(0, 1) },
  ];
  json.Version = 4;

  const salida = path.join(runtime, `${nombre}.casavita.model3.json`);
  fs.writeFileSync(salida, JSON.stringify(json, null, 2));

  console.log(`${nombre}: Version ${json.Version}`);
  console.log(`  EyeBlink -> ${ojos.length ? ojos.join(', ') : '(ninguno)'}`);
  console.log(`  LipSync  -> ${boca.slice(0, 1).join(', ') || '(ninguno)'}`);
  console.log(`  escrito  -> ${salida}`);
}

const nombres = fs
  .readdirSync(RAIZ_MODELOS, { withFileTypes: true })
  .filter((e) => e.isDirectory())
  .map((e) => e.name)
  .filter((n) => fs.existsSync(path.join(RAIZ_MODELOS, n, 'runtime', `${n}.model3.json`)));

for (const n of nombres) parchear(n);
