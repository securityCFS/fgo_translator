import assert from "node:assert/strict";
import { webcrypto } from "node:crypto";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

globalThis.window = globalThis;
if (!globalThis.crypto) {
  Object.defineProperty(globalThis, "crypto", { value: webcrypto });
}
globalThis.localStorage = {
  getItem() {
    return JSON.stringify({
      translationCacheBaseUrl: "https://raw.githubusercontent.com/securityCFS/translator_cache_repo/main",
      translationCacheUploadUrl: "https://worker.example/upload",
    });
  },
};

const apiSource = (await readFile(path.join(root, "js", "api.js"), "utf8"))
  .replace("const AA = (() => {", "globalThis.AA = (() => {");
globalThis.eval(apiSource);

const cacheSource = await readFile(path.join(root, "js", "translation-cache.js"), "utf8");
globalThis.eval(cacheSource);

const indexSource = await readFile(path.join(root, "index.html"), "utf8");
assert.match(indexSource, /id="translationCacheSelect"/);
assert.match(indexSource, /loadTranslationCacheOptions\(\)/);
assert.match(indexSource, /readScriptVersion/);

const rawFixture = [
  "＠キャスター",
  "[%1]の同行者として[r]警戒されている彼女は避けたかった。[k]",
  "＠F：神王ラックランド",
  "……？[r]誰、だ？[k]",
].join("\n");

const fixtureDialogues = globalThis.AA.parseDialogues(rawFixture, "fixture");
assert.equal(fixtureDialogues[0].content, "藤丸立香の同行者として\n警戒されている彼女は避けたかった。");
assert.equal(fixtureDialogues[1].speaker, "F：神王ラックランド");

const originalFetch = globalThis.fetch;
let uploadRequest;
globalThis.fetch = async (url, options) => {
  uploadRequest = {
    url,
    method: options?.method,
    headers: options?.headers,
    body: JSON.parse(options?.body),
  };
  return new Response(JSON.stringify({ ok: true, status: "created", path: "v1/example.json" }), {
    status: 200,
    headers: { "content-type": "application/json" },
  });
};

const uploadResult = await globalThis.TranslationCache.uploadScript({
  scriptId: "9415571810",
  sourceRegion: "JP",
  dialogues: fixtureDialogues,
  targetLanguage: "Chinese (Simplified)",
  apiType: "openai",
  model: "deepseek-v4-flash",
  translations: [
    { speaker: "キャスター", translated_content: "同行者。"},
    { speaker: "F：神王ラックランド", translated_content: "是谁？"},
  ],
});

assert.equal(uploadRequest.url, "https://worker.example/upload");
assert.equal(uploadRequest.method, "POST");
assert.equal(uploadRequest.body.script_id, "9415571810");
assert.equal(uploadRequest.body.source_region, "JP");
assert.equal(uploadRequest.body.target_language, "Chinese (Simplified)");
assert.equal(uploadRequest.body.api_type, "openai");
assert.equal(uploadRequest.body.model, "deepseek-v4-flash");
assert.match(uploadRequest.body.source_hash, /^[a-f0-9]{64}$/);
assert.equal(uploadRequest.body.translations.length, 2);
assert.equal(uploadResult.status, "created");

const uploadCalls = [];
globalThis.fetch = async (url, options) => {
  uploadCalls.push({ url, body: JSON.parse(options?.body) });
  return new Response(JSON.stringify({ ok: true, status: "created" }), { status: 200 });
};
const batchResult = await globalThis.TranslationCache.uploadScripts({
  scriptIds: ["111", "222"],
  scriptDialogueCounts: [1, 1],
  allDialogues: fixtureDialogues,
  translatedDialogues: [
    { speaker: "キャスター", translated_content: "同行者。"},
    { speaker: "F：神王ラックランド", translated_content: "是谁？"},
  ],
  sourceRegion: "JP",
  targetLanguage: "Chinese (Simplified)",
  apiType: "openai",
  model: "deepseek-v4-flash",
});
assert.equal(batchResult.length, 2);
assert.equal(uploadCalls.length, 2);
assert.deepEqual(uploadCalls.map((call) => call.body.script_id), ["111", "222"]);
globalThis.fetch = originalFetch;

const fixtureHash = await globalThis.TranslationCache.canonicalSourceHash(fixtureDialogues);
const optionPayload = {
  schema_version: 1,
  script_id: "1",
  source_region: "JP",
  source_hash: fixtureHash,
  target_language: "zh-CN",
  provider: "deepseek",
  model: "deepseek-v4-flash",
  prompt_version: "fgo-v1",
  dialogue_count: fixtureDialogues.length,
  trusted_generation: true,
  generator: { generated_at: "2026-05-14T00:00:00Z" },
  translations: fixtureDialogues.map((dialogue, index) => ({
    speaker: dialogue.speaker || "",
    translated_content: `cached line ${index + 1}`,
  })),
};

globalThis.fetch = async (url) => {
  const textUrl = String(url);
  const basePath = `/contents/v1/JP/1/${fixtureHash}/zh-CN`;
  if (textUrl.includes(basePath) && !textUrl.includes("/deepseek")) {
    return new Response(JSON.stringify([{ type: "dir", name: "deepseek" }]), { status: 200 });
  }
  if (textUrl.includes(`${basePath}/deepseek`) && !textUrl.includes("/deepseek-v4-flash")) {
    return new Response(JSON.stringify([{ type: "dir", name: "deepseek-v4-flash" }]), { status: 200 });
  }
  if (textUrl.includes(`${basePath}/deepseek/deepseek-v4-flash`)) {
    return new Response(JSON.stringify([
      { type: "file", name: "fgo-v1.json", download_url: "https://raw.example/fgo-v1.json" },
    ]), { status: 200 });
  }
  if (textUrl === "https://raw.example/fgo-v1.json") {
    return new Response(JSON.stringify(optionPayload), { status: 200 });
  }
  if (textUrl.includes(`/v1/JP/1/${fixtureHash}/zh-CN/deepseek/deepseek-v4-flash/fgo-v1.json`)) {
    return new Response(JSON.stringify(optionPayload), { status: 200 });
  }
  return new Response("not found", { status: 404 });
};

const options = await globalThis.TranslationCache.listOptions({
  scriptId: "1",
  sourceRegion: "JP",
  dialogues: fixtureDialogues,
  targetLanguage: "Chinese (Simplified)",
});
assert.equal(options.length, 1);
assert.equal(options[0].label, "deepseek / deepseek-v4-flash / fgo-v1");

const chosenCached = await globalThis.TranslationCache.readScriptVersion({
  scriptId: "1",
  sourceRegion: "JP",
  dialogues: fixtureDialogues,
  targetLanguage: "Chinese (Simplified)",
  provider: "deepseek",
  model: "deepseek-v4-flash",
  promptVersion: "fgo-v1",
});
assert.equal(chosenCached.length, fixtureDialogues.length);
globalThis.fetch = originalFetch;

if (process.env.RUN_LIVE_CACHE_CONTRACT === "1") {
  const { dialogues } = await globalThis.AA.extractDialogues("9415571810", "JP");
  const sourceHash = await globalThis.TranslationCache.canonicalSourceHash(dialogues);
  assert.equal(sourceHash, "3557313fa59c588d7dc69818e91b445d695aa8f13e46f2e48c6bce9c04fe1432");

  const cached = await globalThis.TranslationCache.readScript({
    scriptId: "9415571810",
    sourceRegion: "JP",
    dialogues,
    targetLanguage: "Chinese (Simplified)",
    apiType: "openai",
    model: "deepseek-v4-flash",
  });
  assert.equal(cached.length, 80);
}

console.log("static cache contract ok");
