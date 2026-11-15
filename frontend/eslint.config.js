import js from "@eslint/js";
import globals from "globals";

export default [
  { ignores: ["dist/**"] },
  js.configs.recommended,
  {
    files: ["src/**/*.{js,jsx}"],
    languageOptions: {
      ecmaVersion: "latest",
      sourceType: "module",
      globals: { ...globals.browser, ...globals.node },
      parserOptions: { ecmaFeatures: { jsx: true } },
    },
    rules: {
      // React's JSX runtime references component identifiers outside of plain
      // JavaScript AST reads; uppercase component/icon names are used in JSX.
      "no-unused-vars": ["warn", { argsIgnorePattern: "^_|^[A-Z]", varsIgnorePattern: "^_|^[A-Z]" }],
    },
  },
];
