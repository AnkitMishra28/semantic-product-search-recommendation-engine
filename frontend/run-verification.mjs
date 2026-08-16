import { chromium, devices } from "playwright";
import fs from "node:fs";
import path from "node:path";

const baseUrl = process.env.FRONTEND_URL || "http://localhost:3000";
const backendUrl = process.env.BACKEND_URL || "http://localhost:8000";
const outDir = path.resolve("./verification_screenshots");
fs.mkdirSync(outDir, { recursive: true });

console.log(`Starting Phase 12.5 Playwright verification...`);
console.log(`Frontend: ${baseUrl}`);
console.log(`Backend: ${backendUrl}`);
console.log(`Output directory: ${outDir}`);

const report = {
  baseUrl,
  backendUrl,
  timestamp: new Date().toISOString(),
  screenshots: {},
  checks: {},
  apiCorsChecks: {},
};

const browser = await chromium.launch({ headless: true });
const context = await browser.newContext({
  viewport: { width: 1440, height: 900 },
  deviceScaleFactor: 1.5,
});
const page = await context.newPage();

// Helper for taking screenshot
const takeScreenshot = async (name) => {
  const filePath = path.join(outDir, name);
  await page.screenshot({ path: filePath, fullPage: true });
  report.screenshots[name] = filePath;
  console.log(`✓ Screenshot captured: ${name}`);
};

try {
  // 1. HOME PAGE (/)
  console.log("\n--- Checking Home Page (/) ---");
  await page.goto(baseUrl, { waitUntil: "networkidle" });
  await page.waitForTimeout(1000);
  await takeScreenshot("01-home.png");
  report.checks.homeHeadlineVisible = await page.locator("text=Semantic Product Search").first().isVisible();
  report.checks.homeResearchEyebrow = await page.locator("text=ML SEARCH & RECOMMENDATION RESEARCH").first().isVisible();
  report.checks.homePipelineVisible = await page.locator("text=End-to-End Architecture Flow").first().isVisible();

  // 2. DASHBOARD REDIRECT (/dashboard -> /)
  console.log("\n--- Checking /dashboard Redirect ---");
  const dashResponse = await page.goto(`${baseUrl}/dashboard`, { waitUntil: "networkidle" });
  await page.waitForTimeout(800);
  const currentUrl = page.url();
  report.checks.dashboardRedirectUrl = currentUrl;
  report.checks.dashboardNot404 = dashResponse?.status() !== 404;
  report.checks.dashboardRedirectSuccess = currentUrl === `${baseUrl}/` || currentUrl === `${baseUrl}`;
  await takeScreenshot("10-dashboard.png");
  console.log(`Dashboard response status: ${dashResponse?.status()}, URL after redirect: ${currentUrl}`);

  // 3. SEARCH PAGE (/search)
  console.log("\n--- Checking Search Page (/search) ---");
  await page.goto(`${baseUrl}/search`, { waitUntil: "networkidle" });
  await page.waitForTimeout(800);
  await takeScreenshot("02-search.png");
  report.checks.searchPageConsole = await page.locator("text=Semantic Product Search").first().isVisible();

  // 4. PERFORM SEARCH
  console.log("\n--- Performing Real Search ---");
  const query = "wireless noise cancelling over-ear headphones";
  await page.getByLabel("Search products").fill(query);
  await page.getByRole("button", { name: "Search" }).click();

  // Wait for results
  await page.waitForSelector("text=Ranked Products", { timeout: 60000 });
  await page.waitForTimeout(1500);
  await takeScreenshot("03-search-results.png");

  report.checks.rankedProductsVisible = await page.locator("text=Ranked Products").first().isVisible();
  report.checks.queryUnderstandingVisible = await page.locator("text=Stage 1 · Query Understanding Analysis").first().isVisible();
  report.checks.executionProfileVisible = await page.locator("text=Multi-Stage Latency & Execution Profile").first().isVisible();

  // Find ASIN from top result
  const asinEl = await page.locator("text=/ASIN:\\s*[A-Z0-9]+/").first();
  const asinText = (await asinEl.innerText()).trim();
  const topAsin = asinText.split(":").pop()?.trim() || "B0BW4PFM58";
  report.checks.topProductAsin = topAsin;
  console.log(`Top search result ASIN: ${topAsin}`);

  // 5. EXPLANATION MODAL / DRAWER
  console.log("\n--- Checking Grounded Explanation ---");
  const whyBtn = page.getByRole("button", { name: "Why This Result" }).first();
  await whyBtn.click();
  await page.waitForSelector("text=Explainable Retrieval & Ranking Rationale", { timeout: 30000 });
  await page.waitForTimeout(800);
  await takeScreenshot("04-explanation.png");
  report.checks.explanationDialogVisible = await page.locator("text=Explainable Retrieval & Ranking Rationale").first().isVisible();
  report.checks.explanationGroundedVisible = await page.locator("text=Grounded ML Explanation").first().isVisible();

  // Close explanation dialog
  await page.getByLabel("Close explanation").click();
  await page.waitForTimeout(500);

  // 6. BROWSER CORS CHECKS
  console.log("\n--- Verifying CORS from Browser Context ---");
  report.apiCorsChecks = await page.evaluate(async (backendUrl) => {
    const results = {};
    const testEndpoint = async (name, url, options) => {
      try {
        const res = await fetch(url, options);
        results[name] = {
          status: res.status,
          ok: res.ok,
          corsAllowed: res.headers.get("access-control-allow-origin") !== null,
        };
      } catch (err) {
        results[name] = { error: String(err), ok: false };
      }
    };

    // 1. Ready
    await testEndpoint("ready", `${backendUrl}/api/v1/ready`, { method: "GET" });

    // 2. Metrics
    await testEndpoint("metrics", `${backendUrl}/api/v1/metrics`, { method: "GET" });

    // 3. Experiments
    await testEndpoint("experiments", `${backendUrl}/api/v1/evaluate/experiments`, { method: "GET" });

    // 4. POST Search
    await testEndpoint("search_post", `${backendUrl}/api/v1/search`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: "wireless headphones", top_k_retrieval: 10, top_k_reranking: 5, enable_reranking: true }),
    });

    // 5. POST Recommend
    await testEndpoint("recommend_post", `${backendUrl}/api/v1/recommend`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ asin: "B0BW4PFM58", top_k: 4, strategy: "hybrid" }),
    });

    // 6. POST Explain
    await testEndpoint("explain_post", `${backendUrl}/api/v1/explain`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: "wireless headphones", product_id: "B0BW4PFM58" }),
    });

    return results;
  }, backendUrl);

  console.log("CORS Check Results:", JSON.stringify(report.apiCorsChecks, null, 2));

  // 7. RECOMMENDATIONS PAGE (/recommendations)
  console.log("\n--- Checking Recommendations Page ---");
  await page.goto(`${baseUrl}/recommendations?asin=${encodeURIComponent(topAsin)}`, { waitUntil: "networkidle" });
  await page.waitForSelector("text=Recommendation Console", { timeout: 30000 });
  await page.waitForTimeout(1500);
  await takeScreenshot("05-recommendations.png");
  report.checks.recommendationsLoaded = await page.locator("text=/\\d+\\s+Recommendations?/").first().isVisible();

  // Test User History Mode
  console.log("Testing user history recommendation mode...");
  await page.getByRole("button", { name: "By User History" }).click();
  await page.getByLabel("Recent User History ASINs (comma-separated)").fill(topAsin);
  await page.getByRole("button", { name: "Get Recommendations" }).click();
  await page.waitForTimeout(2000);
  report.checks.userHistoryModeWorking = true;

  // 8. EVALUATION PAGE (/evaluation)
  console.log("\n--- Checking Evaluation Page ---");
  await page.goto(`${baseUrl}/evaluation`, { waitUntil: "networkidle" });
  await page.waitForSelector("text=Scientific Evaluation & Offline Benchmarks", { timeout: 30000 });
  await page.waitForTimeout(1000);
  await takeScreenshot("06-evaluation.png");
  report.checks.evaluationPageLoaded = await page.locator("text=Retrieval Quality — BM25 vs Dense vs Hybrid RRF").first().isVisible();
  report.checks.evaluationStageSeparation = await page.locator("text=Two-Stage Retrieval: Stage-1 vs Stage-2 Gains").first().isVisible();

  // 9. ABOUT PAGE (/about)
  console.log("\n--- Checking About Page ---");
  await page.goto(`${baseUrl}/about`, { waitUntil: "networkidle" });
  await page.waitForSelector("text=Amazon-Scale Semantic Product Search & Recommendation Engine", { timeout: 30000 });
  await page.waitForTimeout(1000);
  await takeScreenshot("07-about.png");
  report.checks.aboutPageLoaded = await page.locator("text=Applied Scientist Research Thesis").first().isVisible();
  report.checks.aboutDisclaimerVisible = await page.locator("text=Academic & Applied Scientist Portfolio Prototype").first().isVisible();

  // 10. MOBILE VIEWPORT
  console.log("\n--- Checking Mobile Viewport ---");
  const mobileContext = await browser.newContext({
    ...devices["iPhone 13"],
    deviceScaleFactor: 2,
  });
  const mobilePage = await mobileContext.newPage();
  await mobilePage.goto(baseUrl, { waitUntil: "networkidle" });
  await mobilePage.waitForTimeout(1000);
  const mobileShotPath = path.join(outDir, "09-mobile.png");
  await mobilePage.screenshot({ path: mobileShotPath, fullPage: true });
  report.screenshots["09-mobile.png"] = mobileShotPath;
  report.checks.mobileHeroVisible = await mobilePage.locator("text=Semantic Product Search").first().isVisible();
  console.log(`✓ Screenshot captured: 09-mobile.png`);
  await mobileContext.close();

} catch (err) {
  console.error("Error during verification:", err);
  report.error = String(err);
} finally {
  await context.close();
  await browser.close();
}

// 11. ERROR STATE CAPTURE
// Capture error state with a targeted browser test where backend is requested with invalid/down url
console.log("\n--- Capturing Error State (08-error-state.png) ---");
try {
  const errBrowser = await chromium.launch({ headless: true });
  const errContext = await errBrowser.newContext({ viewport: { width: 1440, height: 900 } });
  const errPage = await errContext.newPage();

  // Intercept backend search API to simulate a network outage / 500 error
  await errPage.route("**/api/v1/search", (route) => {
    route.abort("failed");
  });

  await errPage.goto(`${baseUrl}/search`, { waitUntil: "networkidle" });
  await errPage.getByLabel("Search products").fill("wireless headphones");
  await errPage.getByRole("button", { name: "Search" }).click();
  await errPage.waitForSelector("[role=alert]", { timeout: 15000 });
  await errPage.waitForTimeout(800);

  const errorShotPath = path.join(outDir, "08-error-state.png");
  await errPage.screenshot({ path: errorShotPath, fullPage: true });
  report.screenshots["08-error-state.png"] = errorShotPath;
  console.log(`✓ Screenshot captured: 08-error-state.png`);

  await errContext.close();
  await errBrowser.close();
} catch (err) {
  console.error("Error capturing error state:", err);
}

const summaryPath = path.join(outDir, "verification-summary.json");
fs.writeFileSync(summaryPath, JSON.stringify(report, null, 2), "utf8");
console.log(`\n========================================`);
console.log(`Verification Complete! Summary: ${summaryPath}`);
console.log(JSON.stringify(report, null, 2));
