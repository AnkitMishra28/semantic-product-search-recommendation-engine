import { chromium } from "playwright";
import path from "node:path";

const outDir = process.env.PHASE125_OUTDIR || path.join(process.env.TEMP || ".", "phase125-verification");
const browser = await chromium.launch({ headless: true });
const context = await browser.newContext({ viewport: { width: 1440, height: 900 } });
const page = await context.newPage();

await page.goto("http://localhost:3000/search", { waitUntil: "domcontentloaded" });
await page.getByRole("button", { name: /wireless noise cancelling over-ear headphones/i }).click();

try {
  await Promise.race([
    page.waitForSelector("text=Backend unavailable", { timeout: 60000 }),
    page.waitForSelector("text=Request timed out", { timeout: 60000 }),
    page.waitForSelector("[role=alert]", { timeout: 60000 }),
  ]);
} catch {}

await page.waitForTimeout(1200);
const outPath = path.join(outDir, "08-error-state.png");
await page.screenshot({ path: outPath, fullPage: true });
console.log(outPath);

await context.close();
await browser.close();
