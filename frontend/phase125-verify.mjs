import { chromium, devices } from "playwright";
import fs from "node:fs";
import path from "node:path";

const baseUrl = "http://localhost:3000";
const backendUrl = "http://localhost:8000";
const outDir = process.env.PHASE125_OUTDIR || path.join(process.env.TEMP || ".", "phase125-verification");
fs.mkdirSync(outDir, { recursive: true });

const result = {
  baseUrl,
  backendUrl,
  screenshots: [],
  checks: {},
  asinUsed: null,
  apiChecks: {},
};

const browser = await chromium.launch({ headless: true });
const context = await browser.newContext({ viewport: { width: 1536, height: 960 } });
const page = await context.newPage();

const shot = async (name) => {
  const p = path.join(outDir, name);
  await page.screenshot({ path: p, fullPage: true });
  result.screenshots.push(p);
};

await page.goto(baseUrl, { waitUntil: "domcontentloaded" });
await page.waitForTimeout(1000);
await shot("01-home.png");
result.checks.homeLoaded = !!(await page.locator("text=Semantic Product Search").first().count());

const dashboardResp = await page.goto(`${baseUrl}/dashboard`, { waitUntil: "domcontentloaded" });
await page.waitForTimeout(700);
result.checks.dashboardStatus = dashboardResp?.status() ?? null;
result.checks.dashboardUrlAfter = page.url();
result.checks.dashboardRedirectedToHome = page.url() === `${baseUrl}/` || page.url() === `${baseUrl}`;

await page.goto(`${baseUrl}/search`, { waitUntil: "domcontentloaded" });
await page.waitForTimeout(800);
await shot("02-search.png");

const query = "wireless noise cancelling over-ear headphones";
await page.getByLabel("Search products").fill(query);
await page.getByRole("button", { name: "Search" }).click();
await page.waitForSelector("text=Ranked Products", { timeout: 120000 });
await page.waitForTimeout(1200);
await shot("03-search-results.png");

result.checks.searchResultsVisible = await page.locator("text=Ranked Products").first().isVisible();
result.checks.queryUnderstandingVisible = await page.locator("text=Query Understanding").first().isVisible();

const asinText = (await page.locator("text=/ASIN:\\s*[A-Z0-9]+/").first().innerText()).trim();
const asin = asinText.split(":").pop()?.trim() || "";
result.asinUsed = asin;

const whyBtn = page.getByRole("button", { name: "Why This Result" }).first();
await whyBtn.click();
await page.waitForSelector("text=Explainable Retrieval & Ranking Rationale", { timeout: 120000 });
await page.waitForTimeout(600);
await shot("04-explanation.png");
result.checks.explanationDialogVisible = await page.locator("text=Explainable Retrieval & Ranking Rationale").first().isVisible();
await page.getByLabel("Close explanation").click();

result.apiChecks = await page.evaluate(async ({ backendUrl, query, asin }) => {
  const json = { "Content-Type": "application/json" };
  const report = {};
  const call = async (name, input) => {
    try {
      const res = await fetch(`${backendUrl}${input.path}`, input.init);
      report[name] = { ok: res.ok, status: res.status };
    } catch (error) {
      report[name] = { ok: false, error: String(error) };
    }
  };

  await call("search", {
    path: "/api/v1/search",
    init: {
      method: "POST",
      headers: json,
      body: JSON.stringify({ query, top_k_retrieval: 50, top_k_reranking: 10, enable_reranking: true }),
    },
  });

  await call("recommend", {
    path: "/api/v1/recommend",
    init: {
      method: "POST",
      headers: json,
      body: JSON.stringify({ asin, top_k: 5, strategy: "hybrid" }),
    },
  });

  await call("explain", {
    path: "/api/v1/explain",
    init: {
      method: "POST",
      headers: json,
      body: JSON.stringify({ query, product_id: asin }),
    },
  });

  await call("metrics", { path: "/api/v1/metrics", init: { method: "GET" } });
  await call("experiments", { path: "/api/v1/evaluate/experiments", init: { method: "GET" } });
  await call("ready", { path: "/api/v1/ready", init: { method: "GET" } });

  return report;
}, { backendUrl, query, asin });

await page.goto(`${baseUrl}/recommendations?asin=${encodeURIComponent(asin)}`, { waitUntil: "domcontentloaded" });
await page.waitForTimeout(1800);
await shot("05-recommendations.png");
result.checks.recommendationsLoaded = await page.locator("text=/Recommendation(s)?/").first().isVisible();

await page.getByRole("button", { name: "By User History" }).click();
await page.getByLabel("Recent User History ASINs (comma-separated)").fill(asin);
await page.getByRole("button", { name: "Get Recommendations" }).click();
await page.waitForTimeout(2200);
result.checks.userHistoryModeExercised = true;

await page.goto(`${baseUrl}/evaluation`, { waitUntil: "domcontentloaded" });
await page.waitForSelector("text=Evaluation & Benchmarks", { timeout: 120000 });
await page.waitForTimeout(700);
await shot("06-evaluation.png");

await page.goto(`${baseUrl}/about`, { waitUntil: "domcontentloaded" });
await page.waitForSelector("text=About This Project", { timeout: 120000 });
await page.waitForTimeout(700);
await shot("07-about.png");

const mobileContext = await browser.newContext({ ...devices["iPhone 13"] });
const mobilePage = await mobileContext.newPage();
await mobilePage.goto(baseUrl, { waitUntil: "domcontentloaded" });
await mobilePage.waitForTimeout(900);
const mobileShot = path.join(outDir, "09-mobile.png");
await mobilePage.screenshot({ path: mobileShot, fullPage: true });
result.screenshots.push(mobileShot);
await mobileContext.close();

await context.close();
await browser.close();

const outJson = path.join(outDir, "verification-main.json");
fs.writeFileSync(outJson, JSON.stringify(result, null, 2), "utf8");
console.log(outJson);
console.log(JSON.stringify(result, null, 2));
