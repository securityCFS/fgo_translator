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
};
vm.createContext(context);
vm.runInContext(
  `${extractFunction('escapeHtml')}\n${extractFunction('formatScriptText')}\n${extractFunction('formatSpeakerName')}\n`
    + `const SCENE_LOGICAL_WIDTH = 1024; const SCENE_LOGICAL_HEIGHT = 576;\n${extractFunction('computeSpriteLayout')}`,
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
assert.equal(explicitLayout.bottomCqh, 12.5);
assert.equal(explicitLayout.scale, 1.25);
assert.equal(explicitLayout.depth, 7);
assert.equal(explicitLayout.explicit, true);

const fallbackLayout = context.computeSpriteLayout({
  entityId: '1002', x: null, y: null,
}, 68, 1);
assert.equal(fallbackLayout.key, '1002:1');
assert.equal(fallbackLayout.leftPercent, 68);
assert.equal(fallbackLayout.bottomCqh, 0);
assert.equal(fallbackLayout.explicit, false);

console.log('gaming formatting tests passed');
