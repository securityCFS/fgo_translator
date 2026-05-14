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
