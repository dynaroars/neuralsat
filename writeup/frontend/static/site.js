// Theme switch: the choice is kept in localStorage; without it the system setting decides.
(function () {
	var button = document.querySelector("nav .theme");
	if (!button) return;
	var root = document.documentElement;
	function dark() {
		var chosen = root.dataset.theme;
		if (chosen) return chosen === "dark";
		return window.matchMedia("(prefers-color-scheme: dark)").matches;
	}
	function show() { button.textContent = dark() ? "☀️" : "🌙"; }
	button.addEventListener("click", function () {
		var next = dark() ? "light" : "dark";
		root.dataset.theme = next;
		try { localStorage.setItem("theme", next); } catch (e) {}
		show();
	});
	show();
})();
