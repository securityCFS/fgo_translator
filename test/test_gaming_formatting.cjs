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
  pendingImages: [],
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
context.Image = class FakeImage {
  set src(value) {
    this.requestedUrl = value;
    context.pendingImages.push(this);
  }
};
vm.createContext(context);
vm.runInContext(
  `${extractFunction('escapeHtml')}\n${extractFunction('formatScriptText')}\n${extractFunction('formatSpeakerName')}\n`
    + `const SCENE_LOGICAL_WIDTH = 1024; const SCENE_LOGICAL_HEIGHT = 576; const DEFAULT_FIGURE_BODY_HEIGHT = 768;\n`
    + `const FACE_DEFAULT = 256; const FIGURE_DEFAULT_H = 1024; const FIGURE_PAGE_W = 1024; const FIGURE_PAGE_H = 1024;\n`
    + `${extractFunction('computeSpriteLayout')}\n${extractFunction('getSubRenderMaskUrl')}\n${extractFunction('computeSubRenderLayout')}\n${extractFunction('getFrameAutoAdvanceDelay')}\n${extractFunction('getFigureBodyUrl')}\n${extractFunction('getFaceCrop')}\n${extractFunction('getFigureViewport')}\n${extractFunction('applySpriteVisualState')}\n${extractFunction('shouldRenderStageSprite')}\n`
    + `let DESIRED_BG_URL = ''; let BG_PENDING_URL = ''; let BG_LOAD_TOKEN = 0;\n${extractFunction('syncBackground')}`,
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
  slot: 'E', entityId: '1049000', x: 600, y: -30, scale: 1, depth: 2,
}, 50, 0);
assert.ok(Math.abs(maskedSubRenderLayout.heightCqh - 133.3333333333) < 1e-8);
assert.equal(maskedSubRenderLayout.leftPercent > 92, true);

const subRenderLayout = context.computeSubRenderLayout({
  visible: true, x: 400, y: -280, scale: 0.8, depth: 6, mask: 'cut359_mask16',
}, 'JP');
assert.equal(
  subRenderLayout.transform,
  'translate(39.0625cqw, 48.61111111111111cqh) scale(0.8)',
);
assert.equal(subRenderLayout.zIndex, 9);
assert.equal(
  subRenderLayout.maskUrl,
  'https://static.atlasacademy.io/JP/Image/cut359_mask16/cut359_mask16.png',
);
assert.equal(context.getFrameAutoAdvanceDelay({ type: 'stage', duration: 0.6 }), 600);

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
}, subRenderFigureLayout);
assert.ok(Math.abs(Number(visualVars.get('--sprite-opacity')) - (0.6 * 128 / 255)) < 1e-8);
assert.equal(visualClasses.get('silhouette'), true);

context.applySpriteVisualState(visualNode, {
  opacity: 0.6, filter: 'normal', talking: false,
}, maskedSubRenderLayout);
assert.equal(visualVars.get('--sprite-opacity'), '0.6');
assert.equal(visualVars.get('--sprite-dim-opacity'), '0.6');
assert.equal(visualClasses.get('dimmed'), true);

assert.equal(
  context.getFigureBodyUrl('https://static.atlasacademy.io/JP/CharaFigure/1/1_merged.png'),
  'https://static.atlasacademy.io/JP/CharaFigure/1/1.png',
);
assert.deepEqual(
  JSON.parse(JSON.stringify(context.getFaceCrop(1, 256, 256, 1024))),
  { srcX: 0, srcY: 768 },
);
assert.deepEqual(
  JSON.parse(JSON.stringify(context.getFaceCrop(18, 256, 447, 1024))),
  { srcX: 256, srcY: 3072 },
);
assert.deepEqual(
  JSON.parse(JSON.stringify(context.getFaceCrop(9, 256, 447, 1024))),
  { srcX: 0, srcY: 2048 },
);
assert.deepEqual(
  JSON.parse(JSON.stringify(context.getFaceCrop(16, 256, 447, 1024))),
  { srcX: 768, srcY: 2495 },
);
assert.deepEqual(
  JSON.parse(JSON.stringify(context.getFigureViewport(2048, 1024, 126))),
  { cropX: 512, width: 1024, height: 702 },
);
assert.deepEqual(
  JSON.parse(JSON.stringify(context.getFigureViewport(1024, 768, 143))),
  { cropX: 0, width: 1024, height: 719 },
);
assert.match(html, /if \(assetType !== 'chara'\)/);
assert.match(html, /COMPOSITE_CACHE\[key\] = sp\.url/);

const bgClasses = new Set();
const bg = {
  src: 'https://example.test/A.png',
  classList: {
    add: value => bgClasses.add(value),
    remove: value => bgClasses.delete(value),
  },
};
context.syncBackground(bg, 'https://example.test/B.png');
assert.equal(bgClasses.has('fading'), true);
const staleRequest = context.pendingImages.shift();
context.syncBackground(bg, 'https://example.test/A.png');
assert.equal(bgClasses.has('fading'), false);
staleRequest.onload();
assert.equal(bg.src, 'https://example.test/A.png');
assert.equal(bgClasses.has('fading'), false);

const mainBackground = 'https://static.atlasacademy.io/JP/Back/back105500_1344_626.png';
assert.equal(context.shouldRenderStageSprite({ assetType: 'scene', url: mainBackground }, mainBackground), true);
assert.equal(context.shouldRenderStageSprite({ assetType: 'scene', url: 'https://example.test/other.png' }, mainBackground), true);
assert.equal(context.shouldRenderStageSprite({ assetType: 'image', url: mainBackground }, mainBackground), true);

const sceneLayerLayout = context.computeSpriteLayout({
  slot: 'V', entityId: 'scene:105500', assetType: 'scene',
  x: -50, y: -270, scale: 1.2, depth: 12,
}, 50, 0);
assert.equal(sceneLayerLayout.fullStage, true);
assert.equal(sceneLayerLayout.leftPercent, 0);
assert.equal(sceneLayerLayout.bottomCqh, 0);
assert.equal(sceneLayerLayout.heightCqh, 100);
assert.equal(sceneLayerLayout.widthCqw, 100);
assert.match(html, /node\.lastCompositeKey !== key/);
assert.doesNotMatch(html, /IDX !== myFrameIdx/);
assert.doesNotMatch(html, /node\.img\.style\.opacity = '0\.7'/);

const preloadContext = {
  FRAMES: [
    {
      bg: 'background.png',
      sprites: [{ entityId: '1098348500', url: 'figure-merged.png', assetType: 'chara', face: 0 }],
      subRenders: { '#A': { mask: 'cut359_mask16' } },
    },
    { sprites: [{ entityId: '1098348500', url: 'figure-merged.png', assetType: 'chara', face: 3 }] },
    { sprites: [{ entityId: 'scene:105500', url: 'background.png', assetType: 'scene', face: 0 }] },
  ],
};
vm.createContext(preloadContext);
vm.runInContext(
  `${extractFunction('getFigureBodyUrl')}\n${extractFunction('getSubRenderMaskUrl')}\n${extractFunction('collectSceneAssetRequests')}`,
  preloadContext,
);
const preloadRequests = JSON.parse(JSON.stringify(
  preloadContext.collectSceneAssetRequests(preloadContext.FRAMES, 'JP'),
));
assert.deepEqual(preloadRequests, [
  { kind: 'image', url: 'background.png' },
  {
    kind: 'body', key: 'body:figure-merged.png',
    entityId: '1098348500', url: 'figure-merged.png',
  },
  {
    kind: 'image',
    url: 'https://static.atlasacademy.io/JP/Image/cut359_mask16/cut359_mask16.png',
  },
  {
    kind: 'atlas', key: 'atlas:figure-merged.png',
    entityId: '1098348500', url: 'figure-merged.png',
  },
]);
assert.match(html, /const preloadPromise = preloadAllSceneAssets/);
assert.match(html, /await preloadPromise;[\s\S]{0,600}renderFrame\(\);/);
assert.doesNotMatch(html, /preloadUpcomingSpriteAssets/);

console.log('gaming formatting tests passed');
