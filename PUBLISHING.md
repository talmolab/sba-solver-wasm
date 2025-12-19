# Publishing to npm

This package is published to npm as `@talmolab/sba-solver-wasm` and automatically available via jsDelivr CDN.

## Setup (One-Time)

### 1. npm Account & Organization

1. Create an npm account at https://www.npmjs.com/signup
2. Create the `@talmolab` organization at https://www.npmjs.com/org/create

### 2. Initial Publish

The first publish must be done locally to create the package on npm:

```bash
# Login to npm
npm login

# Build the package
npm run build

# Publish (--access public required for scoped packages)
# --auth-type=web opens browser for passkey/WebAuthn authentication
cd pkg && npm publish --access public --auth-type=web && cd ..
```

### 3. Configure OIDC Trusted Publisher

After the package exists on npm, configure OIDC for token-free CI/CD publishing:

1. Go to: https://www.npmjs.com/package/@talmolab/sba-solver-wasm/access
2. Scroll to **"Trusted Publisher"** section
3. Click **"Add trusted publisher"** → **"GitHub Actions"**
4. Fill in:
   - **Owner:** `talmolab`
   - **Repository:** `sba-solver-wasm`
   - **Workflow filename:** `release.yml`
   - **Environment:** *(leave blank)*
5. Click **"Add"**

## Publishing a New Release

Once setup is complete, publishing is automated via GitHub releases:

```bash
# Update version in Cargo.toml, then:
gh release create v0.2.0 --title "v0.2.0" --notes "Release notes here"
```

The GitHub Actions workflow will automatically:
1. Build the WASM package
2. Publish to npm via OIDC (no tokens needed)
3. Include provenance attestations

## CDN URLs

After publishing, the package is available via jsDelivr:

```
https://cdn.jsdelivr.net/npm/@talmolab/sba-solver-wasm@VERSION/sba_solver_wasm.js
https://cdn.jsdelivr.net/npm/@talmolab/sba-solver-wasm@VERSION/sba_solver_wasm_bg.wasm
```

Use `@latest` for the most recent version, or pin to a specific version like `@0.1.0`.

## Manual Publishing

If you need to publish manually (e.g., CI is broken):

```bash
npm login
npm run build
cd pkg && npm publish --access public --auth-type=web && cd ..
```

## Troubleshooting

### OIDC Authentication Fails
- Verify the workflow filename matches exactly (case-sensitive, including `.yml`)
- Ensure you're using GitHub-hosted runners (self-hosted not supported)
- Check that npm CLI is v11.5.1 or later

### Package Name Conflicts
The package is scoped to `@talmolab`, so there should be no conflicts. The scope is set via `--scope talmolab` in the build scripts.
