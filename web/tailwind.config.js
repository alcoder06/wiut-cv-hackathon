// Tailwind build for the website: npx tailwindcss@3 -c web/tailwind.config.js -i web/tailwind.input.css -o web/static/tw.css --minify
module.exports = {
  content: [`${__dirname}/static/index.html`, `${__dirname}/static/app.js`],
  theme: {
    extend: {
      colors: {
        brutLime: "#CCFF00", brutCoral: "#FF5733", brutBlue: "#0055FF", brutYellow: "#FFDE00", brutPink: "#FF90E8",
        brutBg: "#F4F4F0", brutCard: "#FFFFFF", brutDark: "#121212",
      },
      boxShadow: { brut: "4px 4px 0px #000000", "brut-lg": "6px 6px 0px #000000", "brut-sm": "2px 2px 0px #000000" },
      fontFamily: { sans: ["Inter", "sans-serif"], mono: ["JetBrains Mono", "monospace"] },
    },
  },
};
