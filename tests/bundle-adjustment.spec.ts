import { test, expect } from '@playwright/test';

test.describe('WASM Bundle Adjustment', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/examples/');
  });

  test('loads WASM module successfully', async ({ page }) => {
    // Wait for the status to change from "Loading..."
    await expect(page.locator('#status')).not.toContainText('Loading', { timeout: 30000 });

    // Should show success message
    await expect(page.locator('#status')).toContainText('loaded successfully');
    await expect(page.locator('#status')).toHaveClass(/success/);
  });

  test('run test button is enabled after WASM loads', async ({ page }) => {
    // Wait for WASM to load
    await expect(page.locator('#status')).toContainText('loaded successfully', { timeout: 30000 });

    // Check buttons are enabled
    await expect(page.locator('#runTest')).toBeEnabled();
    await expect(page.locator('#loadExample')).toBeEnabled();
    await expect(page.locator('#runCustom')).toBeEnabled();
  });

  test('runs test optimization and shows results', async ({ page }) => {
    // Wait for WASM to load
    await expect(page.locator('#status')).toContainText('loaded successfully', { timeout: 30000 });

    // Click run test button
    await page.locator('#runTest').click();

    // Wait for optimization to complete
    await expect(page.locator('#status')).toContainText('complete', { timeout: 30000 });

    // Check stats are displayed
    await expect(page.locator('#stats')).toBeVisible();
    await expect(page.locator('#initialCost')).not.toHaveText('-');
    await expect(page.locator('#finalCost')).not.toHaveText('-');
    await expect(page.locator('#iterations')).not.toHaveText('-');
    await expect(page.locator('#converged')).not.toHaveText('-');
    await expect(page.locator('#time')).not.toHaveText('-');

    // Check result JSON is populated
    const resultText = await page.locator('#result').textContent();
    expect(resultText).toContain('cameras');
    expect(resultText).toContain('points');
    expect(resultText).toContain('final_cost');
  });

  test('optimization reduces cost', async ({ page }) => {
    // Wait for WASM to load
    await expect(page.locator('#status')).toContainText('loaded successfully', { timeout: 30000 });

    // Run optimization
    await page.locator('#runTest').click();
    await expect(page.locator('#status')).toContainText('complete', { timeout: 30000 });

    // Parse costs from the display
    const initialCostText = await page.locator('#initialCost').textContent();
    const finalCostText = await page.locator('#finalCost').textContent();

    const initialCost = parseFloat(initialCostText || '0');
    const finalCost = parseFloat(finalCostText || '0');

    // Final cost should be less than or equal to initial cost
    expect(finalCost).toBeLessThanOrEqual(initialCost);
  });

  test('optimization converges', async ({ page }) => {
    // Wait for WASM to load
    await expect(page.locator('#status')).toContainText('loaded successfully', { timeout: 30000 });

    // Run optimization
    await page.locator('#runTest').click();
    await expect(page.locator('#status')).toContainText('complete', { timeout: 30000 });

    // Check convergence
    const convergedText = await page.locator('#converged').textContent();
    expect(convergedText).toBe('Yes');
  });

  test('load example populates input fields', async ({ page }) => {
    // Wait for WASM to load
    await expect(page.locator('#status')).toContainText('loaded successfully', { timeout: 30000 });

    // Click load example
    await page.locator('#loadExample').click();

    // Check that textareas are populated
    const camerasValue = await page.locator('#camerasInput').inputValue();
    const pointsValue = await page.locator('#pointsInput').inputValue();
    const observationsValue = await page.locator('#observationsInput').inputValue();

    expect(camerasValue.length).toBeGreaterThan(10);
    expect(pointsValue.length).toBeGreaterThan(10);
    expect(observationsValue.length).toBeGreaterThan(10);

    // Verify it's valid JSON
    expect(() => JSON.parse(camerasValue)).not.toThrow();
    expect(() => JSON.parse(pointsValue)).not.toThrow();
    expect(() => JSON.parse(observationsValue)).not.toThrow();
  });

  test('custom optimization works with loaded example', async ({ page }) => {
    // Wait for WASM to load
    await expect(page.locator('#status')).toContainText('loaded successfully', { timeout: 30000 });

    // Load example data
    await page.locator('#loadExample').click();

    // Run custom optimization
    await page.locator('#runCustom').click();

    // Wait for completion
    await expect(page.locator('#status')).toContainText('complete', { timeout: 30000 });

    // Verify results
    await expect(page.locator('#stats')).toBeVisible();
    const convergedText = await page.locator('#converged').textContent();
    expect(convergedText).toBe('Yes');
  });

  test('handles invalid JSON gracefully', async ({ page }) => {
    // Wait for WASM to load
    await expect(page.locator('#status')).toContainText('loaded successfully', { timeout: 30000 });

    // Enter invalid JSON
    await page.locator('#camerasInput').fill('not valid json');
    await page.locator('#pointsInput').fill('[]');
    await page.locator('#observationsInput').fill('[]');

    // Try to run
    await page.locator('#runCustom').click();

    // Should show error
    await expect(page.locator('#status')).toContainText('Invalid JSON', { timeout: 5000 });
    await expect(page.locator('#status')).toHaveClass(/error/);
  });

  test('solver configuration affects optimization', async ({ page }) => {
    // Wait for WASM to load
    await expect(page.locator('#status')).toContainText('loaded successfully', { timeout: 30000 });

    // Set max iterations to 1
    await page.locator('#maxIterations').fill('1');

    // Run optimization
    await page.locator('#runTest').click();
    await expect(page.locator('#status')).toContainText('complete', { timeout: 30000 });

    // Check iterations is 1
    const iterationsText = await page.locator('#iterations').textContent();
    expect(parseInt(iterationsText || '0')).toBeLessThanOrEqual(2); // May be 1 or 2 due to initial eval
  });
});

test.describe('WASM Module API', () => {
  test('WasmBundleAdjuster API works correctly', async ({ page }) => {
    await page.goto('/examples/');

    // Wait for WASM to load
    await expect(page.locator('#status')).toContainText('loaded successfully', { timeout: 30000 });

    // Test the API directly via page.evaluate
    const result = await page.evaluate(async () => {
      // @ts-ignore - module is loaded globally
      const { WasmBundleAdjuster } = await import('/pkg/sba_solver_wasm.js');

      const ba = new WasmBundleAdjuster();

      // Set up a simple test case
      const cameras = [{
        rotation: [1.0, 0.0, 0.0, 0.0],
        translation: [0.0, 0.0, 0.0],
        focal: [500.0, 500.0],
        principal: [320.0, 240.0],
        distortion: [0.0, 0.0, 0.0, 0.0, 0.0]
      }];

      const points = [
        [0.0, 0.0, 5.0],
        [0.1, 0.0, 5.0]
      ];

      const observations = [
        { camera_idx: 0, point_idx: 0, x: 320.0, y: 240.0 },
        { camera_idx: 0, point_idx: 1, x: 330.0, y: 240.0 }
      ];

      ba.set_cameras(JSON.stringify(cameras));
      ba.set_points(JSON.stringify(points));
      ba.set_observations(JSON.stringify(observations));

      return {
        numCameras: ba.num_cameras(),
        numPoints: ba.num_points(),
        numObservations: ba.num_observations()
      };
    });

    expect(result.numCameras).toBe(1);
    expect(result.numPoints).toBe(2);
    expect(result.numObservations).toBe(2);
  });

  test('optimization returns valid result structure', async ({ page }) => {
    await page.goto('/examples/');

    // Wait for WASM to load
    await expect(page.locator('#status')).toContainText('loaded successfully', { timeout: 30000 });

    const result = await page.evaluate(async () => {
      // @ts-ignore
      const { WasmBundleAdjuster } = await import('/pkg/sba_solver_wasm.js');

      const ba = new WasmBundleAdjuster();

      const cameras = [{
        rotation: [1.0, 0.0, 0.0, 0.0],
        translation: [0.0, 0.0, 0.0],
        focal: [500.0, 500.0],
        principal: [320.0, 240.0],
        distortion: [0.0, 0.0, 0.0, 0.0, 0.0]
      }];

      const points = [[0.0, 0.0, 5.0]];

      const observations = [
        { camera_idx: 0, point_idx: 0, x: 320.0, y: 240.0 }
      ];

      ba.set_cameras(JSON.stringify(cameras));
      ba.set_points(JSON.stringify(points));
      ba.set_observations(JSON.stringify(observations));
      ba.set_config(JSON.stringify({
        max_iterations: 10,
        cost_tolerance: 1e-6,
        parameter_tolerance: 1e-8,
        gradient_tolerance: 1e-10,
        robust_loss: "none",
        robust_loss_param: 1.0,
        optimize_extrinsics: true,
        optimize_points: true
      }));

      const resultJson = ba.optimize();
      return JSON.parse(resultJson);
    });

    // Check result structure
    expect(result).toHaveProperty('cameras');
    expect(result).toHaveProperty('points');
    expect(result).toHaveProperty('initial_cost');
    expect(result).toHaveProperty('final_cost');
    expect(result).toHaveProperty('iterations');
    expect(result).toHaveProperty('converged');
    expect(result).toHaveProperty('status');

    // Check types
    expect(Array.isArray(result.cameras)).toBe(true);
    expect(Array.isArray(result.points)).toBe(true);
    expect(typeof result.initial_cost).toBe('number');
    expect(typeof result.final_cost).toBe('number');
    expect(typeof result.iterations).toBe('number');
    expect(typeof result.converged).toBe('boolean');
  });
});
