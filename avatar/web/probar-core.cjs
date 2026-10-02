const fs = require('fs');
const path = require('path');

const RUTA_CORE = path.join(__dirname, 'public', 'live2dcubismcore.min.js');
const RUTA_MOC = '/home/USUARIO/vtuber/modelos/koharu/runtime/koharu.moc3';

async function esperarListo(core, ms = 8000) {
  const limite = Date.now() + ms;
  while (Date.now() < limite) {
    if (core && core.Moc && core.Model) return core;
    await new Promise((r) => setTimeout(r, 100));
  }
  throw new Error('el core no arranco');
}

(async () => {
  const fabrica = require(RUTA_CORE);
  let core = typeof fabrica === 'function' ? fabrica() : fabrica;
  if (core && typeof core.then === 'function') core = await core;
  const listo = await esperarListo(core);

  const bytes = new Uint8Array(fs.readFileSync(RUTA_MOC));
  console.log('moc bytes:', bytes.length);
  console.log('firma:', String.fromCharCode(...bytes.slice(0, 4)));

  const moc = listo.Moc.fromArray(bytes);
  console.log('Moc creado. version moc:', listo.Version.csmGetLatestMocVersion());

  const modelo = moc.createModel();
  const claves = Object.keys(modelo);
  console.log('claves del cdmModel:', claves.join(', '));

  for (const clave of ['drawables', 'drawableOrders', 'drawablesDynamicFlags', 'moc', 'flagModelUpdated']) {
    console.log(`  ${clave}: ${modelo[clave] === undefined ? 'INEXISTENTE' : 'OK -> ' + typeof modelo[clave]}`);
  }

  if (modelo.drawables) {
    console.log('  drawables.count =', modelo.drawables.count);
    console.log('  ids[0] =', modelo.drawables.ids && modelo.drawables.ids[0]);
  }

  console.log('metodos del modelo:', claves.filter((k) => typeof modelo[k] === 'function').slice(0, 30).join(', '));

  const n = modelo.parameterCount;
  console.log('parameterCount =', n, ' parameterDefaultValues =', modelo.parameterDefaultValues ? modelo.parameterDefaultValues.length : 'n/d');
  if (n) {
    const nombres = [];
    for (let i = 0; i < n; i++) nombres.push(modelo.parameterIds[i]);
    console.log('primeros 12 parametros:', nombres.slice(0, 12).join(', '));
    console.log('PARAM_MOUTH_OPEN_Y esta?', nombres.includes('PARAM_MOUTH_OPEN_Y'));
    console.log('PARAM_EYE_L_OPEN esta?', nombres.includes('PARAM_EYE_L_OPEN'));
  }
  process.exit(0);
})().catch((e) => {
  console.error('FALLO:', e && e.stack ? e.stack : e);
  process.exit(1);
});
