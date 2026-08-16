import { chromium, devices } from "playwright";
import fs from "node:fs";
import path from "node:path";

const baseUrl = process.env.FRONTEND_URL || "http://localhost:3000";
const backendUrl = process.env.BACKEND_URL || "http://localhost:8000";
const outDir = path.resolve("./verification_screenshots");
fs.mkdirSync(outDir, { recursive: true });

console.log(`Starting Phase 13 Playwright verification...`);
console.log(`Frontend: ${baseUrl}`);
console.log(`Backend: ${backendUrl}`);

const report = {
  baseUrl,
  backendUrl,
  timestamp: new Date().toISOString(),
  phase: "Phase 13 — Scientific Evaluation & Experiment Analysis Dashboard",
  screenshots: {},
  checks: {},
};

const browser = await chromium.launch({ headless: true });
const context = await browser.newContext({
  viewport: { width: 1440, height: 950 },
  deviceScaleFactor: 1.5,
});
const page = await context.newPage();

const takeScreenshot = async (name) => {
  const filePath = path.join(outDir, name);
  await page.screenshot({ path: filePath, fullPage: true });
  report.screenshots[name] = filePath;
  console.log(`✓ Screenshot captured: ${name}`);
};

try {
  // 1. OPEN EVALUATION DASHBOARD
  console.log("\n--- Checking Evaluation Dashboard (/evaluation) ---");
  await page.goto(`${baseUrl}/evaluation`, { waitUntil: "networkidle" });
  await page.waitForSelector("text=Scientific Evaluation & Experiment Analysis Dashboard", { timeout: 30000 });
  await page.waitForTimeout(1000);
  await takeScreenshot("11-evaluation-overview.png");
  report.checks.dashboardHeaderVisible = true;
  report.checks.kpiCardsVisible = await page.locator("text=Stage-1 Recall@100").first().isVisible();
  report.checks.stageSeparationVisible = await page.locator("text=Stage-1 Candidate Recall@100").first().isVisible();
  report.checks.candidateCoverageVennVisible = await page.locator("text=Candidate Pool Coverage & Dual-Track Recovery Analysis").first().isVisible();

  // 2. RETRIEVAL & NEURAL RANKING TAB
  console.log("\n--- Checking Retrieval & Neural Ranking Tab ---");
  await page.getByRole("button", { name: /Retrieval & Neural Ranking/i }).click();
  await page.waitForTimeout(600);
  // Click metric switcher: MRR@10
  await page.getByRole("button", { name: /MRR@10/i }).first().click();
  await page.waitForTimeout(500);
  await takeScreenshot("12-evaluation-retrieval.png");
  report.checks.retrievalBarChartVisible = await page.locator("text=Pipeline Comparison").first().isVisible();

  // 3. PARAMETER SWEEPS & ABLATIONS TAB
  console.log("\n--- Checking Parameter Sweeps Tab ---");
  await page.getByRole("button", { name: "Parameter Sweeps & Ablations" }).click();
  await page.waitForTimeout(600);
  await takeScreenshot("13-evaluation-sweeps.png");
  report.checks.rrfSweepVisible = await page.locator("text=RRF Smoothing Parameter (k) Ablation Study").first().isVisible();
  report.checks.mmrSweepVisible = await page.locator("text=Maximal Marginal Relevance (MMR) λ Parameter Sweep").first().isVisible();

  // 4. RECOMMENDATION STRATEGIES TAB
  console.log("\n--- Checking Recommendation Strategies Tab ---");
  await page.getByRole("button", { name: "Recommendation Strategies" }).click();
  await page.waitForTimeout(600);
  await takeScreenshot("14-evaluation-recommendations.png");
  report.checks.recommendationBenchmarkVisible = await page.locator("text=Recommendation Strategies Offline Benchmark").first().isVisible();

  // 5. OFFLINE LATENCY TAB
  console.log("\n--- Checking Offline Latency Tab ---");
  await page.getByRole("button", { name: "Offline Latency Profiling" }).click();
  await page.waitForTimeout(600);
  await takeScreenshot("15-evaluation-latency.png");
  report.checks.latencyCardsVisible = await page.locator("text=Dense First-Stage FAISS Retrieval").first().isVisible();

  // 6. EXPERIMENT REGISTRY TAB & SEARCH
  console.log("\n--- Checking Experiment Artifact Registry Tab ---");
  await page.getByRole("button", { name: /Experiment Registry/ }).click();
  await page.waitForTimeout(600);
  await takeScreenshot("16-evaluation-registry.png");
  report.checks.registryTableVisible = await page.locator("text=track_e_hybrid_bm25_faiss_rrf").first().isVisible();

  // 7. EXPERIMENT INSPECTOR MODAL
  console.log("\n--- Inspecting Experiment Artifact Modal ---");
  const inspectBtn = page.getByRole("button", { name: "Inspect" }).first();
  await inspectBtn.click();
  await page.waitForSelector("role=dialog", { timeout: 15000 });
  await page.waitForTimeout(800);
  await takeScreenshot("17-experiment-inspector-modal.png");
  report.checks.experimentModalVisible = await page.locator("role=dialog").first().isVisible();
  report.checks.copyJsonButtonVisible = await page.getByRole("button", { name: /Copy JSON/ }).isVisible();

  // Close modal
  await page.getByRole("button", { name: "Close", exact: true }).click();
  await page.waitForTimeout(400);

} catch (err) {
  console.error("Error during Phase 13 verification:", err);
  report.error = String(err);
} finally {
  await context.close();
  await browser.close();
}

const summaryPath = path.join(outDir, "phase13-verification-summary.json");
fs.writeFileSync(summaryPath, JSON.stringify(report, null, 2), "utf8");
console.log(`\n========================================`);
console.log(`Phase 13 Verification Complete! Summary: ${summaryPath}`);
console.log(JSON.stringify(report, null, 2));
