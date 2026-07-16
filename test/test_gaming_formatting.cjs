const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const html = fs.readFileSync(path.join(__dirname, '..', 'templates', 'gaming.html'), 'utf8');

function extractFunction(name) {
  const marker = `function ${name}`;
  const start = html.indexOf(marker);
  if (start < 0) throw new Error(`${name} not found`);
  const brace = html.indexOf('{', start);
  let depth = 0;
  for (let i = brace; i < html.length; i += 1) {
    const ch = html[i];
    if (ch === '{') depth += 1;
    if (ch === '}') {
      depth -= 1;
      if (depth === 0) return html.slice(start, i + 1);
    }
  }
  throw new Error(`${name} body not closed`);
}

const context = {
  window: { DATA: { region: 'JP' } },
  console,
  SVT: {
    '1098123000': { offsetX: 3, offsetY: 143, extendData: {} },
    '1098330800': { offsetX: -16, offsetY: 126, extendData: {} },
    '1049000': { offsetX: 1, offsetY: 147, extendData: {} },
  },
  IMG_SIZE: {
    '1098123000': { width: 1024, height: 768 },
    '1098330800': { width: 2048, height: 768 },
    '1049000': { width: 1024, height: 768 },
  },
};
vm.createContext(context);
vm.runInContext(
  `${extractFunction('escapeHtml')}\n${extractFunction('formatScriptText')}\n${extractFunction('formatSpeakerName')}\n`
    + `const SCENE_LOGICAL_WIDTH = 1024; const SCENE_LOGICAL_HEIGHT = 576; const DEFAULT_FIGURE_BODY_HEIGHT = 768;\n`
    + `${extractFunction('computeSpriteLayout')}\n${extractFunction('applySpriteVisualState')}`,
  context
);

const colored = context.formatScriptText('[51d4ff]广播语音[-]');
assert.match(colored, /color:#51d4ff/i);
assert.doesNotMatch(colored, /\[51d4ff\]|\[-\]/);
assert.match(colored, />广播语音</);

const nested = context.formatScriptText('[51d4ff]第一行\n第二行[-]');
assert.match(nested, /第一行<br>第二行/);
assert.match(nested, /<\/span>$/);

const ruby = context.formatScriptText('[#彼:か]の王');
assert.match(ruby, /<ruby>彼<rt>か<\/rt><\/ruby>の王/);

const speaker = context.formatSpeakerName('[51d4ff]广播语音[-]');
assert.match(speaker, /color:#51d4ff/i);
assert.doesNotMatch(speaker, /\[51d4ff\]|\[-\]/);

const explicitLayout = context.computeSpriteLayout({
  slot: 'A', entityId: '1001', x: -256, y: 72, scale: 1.25, depth: 7,
}, 50, 0);
assert.equal(explicitLayout.key, 'A');
assert.equal(explicitLayout.leftPercent, 25);
assert.ok(Math.abs(explicitLayout.bottomCqh - (-54.1666666667)) < 1e-8);
assert.ok(Math.abs(explicitLayout.heightCqh - 166.6666666667) < 1e-8);
assert.equal(explicitLayout.scale, 1.25);
assert.equal(explicitLayout.depth, 7);
assert.equal(explicitLayout.explicit, true);

const fallbackLayout = context.computeSpriteLayout({
  entityId: '1002', x: null, y: null,
}, 68, 1);
assert.equal(fallbackLayout.key, '1002:1');
assert.equal(fallbackLayout.leftPercent, 68);
assert.ok(Math.abs(fallbackLayout.bottomCqh - (-33.3333333333)) < 1e-8);
assert.ok(Math.abs(fallbackLayout.heightCqh - 133.3333333333) < 1e-8);
assert.equal(fallbackLayout.explicit, false);

const standardFigureLayout = context.computeSpriteLayout({
  slot: 'A', entityId: '1098123000', x: 0, y: 0, scale: 1, depth: 0,
}, 50, 0);
assert.ok(Math.abs(standardFigureLayout.leftPercent - 50.29296875) < 1e-8);
assert.ok(Math.abs(standardFigureLayout.bottomCqh - (-8.5069444444)) < 1e-8);
assert.ok(Math.abs(standardFigureLayout.heightCqh - 133.3333333333) < 1e-8);

const highResolutionLayout = context.computeSpriteLayout({
  slot: 'H', entityId: '1098330800', x: 0, y: 0, scale: 1, depth: 0,
}, 50, 0);
assert.ok(Math.abs(highResolutionLayout.leftPercent - 48.4375) < 1e-8);
assert.ok(Math.abs(highResolutionLayout.bottomCqh - (-11.4583333333)) < 1e-8);
assert.ok(Math.abs(highResolutionLayout.heightCqh - 133.3333333333) < 1e-8);

const subRenderFigureLayout = context.computeSpriteLayout({
  slot: 'E', entityId: '1049000', x: 350, y: -30, scale: 1, depth: 5,
}, 50, 0);
assert.ok(Math.abs(subRenderFigureLayout.leftPercent - 84.27734375) < 1e-8);
assert.ok(Math.abs(subRenderFigureLayout.bottomCqh - (-13.0208333333)) < 1e-8);
assert.ok(Math.abs(subRenderFigureLayout.heightCqh - 133.3333333333) < 1e-8);

const maskedSubRenderLayout = context.computeSpriteLayout({
  slot: 'E', entityId: '1049000', x: 350, y: -30, scale: 1, depth: 2,
  subCameraMask: 'cut359_mask16',
}, 50, 0);
assert.ok(Math.abs(maskedSubRenderLayout.heightCqh - 100) < 1e-8);
assert.equal(maskedSubRenderLayout.maskedSubCamera, true);
assert.ok(maskedSubRenderLayout.leftPercent <= 92);

const visualVars = new Map();
const visualClasses = new Map();
const visualNode = {
  wrap: {
    style: {
      setProperty: (key, value) => visualVars.set(key, value),
      removeProperty: key => visualVars.delete(key),
    },
    classList: {
      toggle: (key, value) => visualClasses.set(key, value),
    },
  },
};
context.applySpriteVisualState(visualNode, {
  opacity: 0.6, filter: 'silhouette', filterColor: '#000000', filterAlpha: 128 / 255,
  talking: true,
}, maskedSubRenderLayout);
assert.ok(Math.abs(Number(visualVars.get('--sprite-opacity')) - (0.6 * 128 / 255)) < 1e-8);
assert.equal(visualClasses.get('silhouette'), true);
assert.equal(visualClasses.get('masked-sub-camera'), true);

console.log('gaming formatting tests passed');
