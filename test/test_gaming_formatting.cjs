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
  `${extractFunction('escapeHtml')}\n${extractFunction('formatScriptText')}\n${extractFunction('formatSpeakerName')}`,
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

console.log('gaming formatting tests passed');
