document.addEventListener("DOMContentLoaded", function () {
  // Locomotive and ScrollTrigger setup
  loco();

  const box = document.getElementById("img");
  const input = document.getElementById("fileinput");
  const pre = document.getElementById("preview");
  const removeBtn = document.getElementById("removeBtn");
  const button = document.getElementById("input_btn");
  const h4 = document.getElementById("h4");

  if (!input || !pre || !removeBtn) {
    console.error("Missing required DOM elements");
    return;
  }

  input.addEventListener("change", function () {
    const file = this.files[0];
    if (file && file.type.startsWith("image/")) {
      const url = URL.createObjectURL(file);
      box.style.display = "block";
      pre.src = url;
      pre.style.display = "block";
      pre.style.opacity = 1;
      pre.style.transition = "opacity 0.5s ease";
      button.style.opacity = 0;
      h4.style.opacity = 0;
      removeBtn.style.display = "inline-block";
      removeBtn.style.opacity = 1;
    } else {
      alert("Please select a valid image file.");
    }
  });

  removeBtn.addEventListener("click", function () {
    pre.src = "";
    pre.style.transition = "opacity 0.9s ease";
    pre.style.opacity = 0;
    box.style.display = "none";
    removeBtn.style.opacity = 0;

    setTimeout(() => {
      pre.style.display = "none";
      pre.style.opacity = 1;
      removeBtn.style.display = "none";
      removeBtn.style.opacity = 1;
    }, 900);

    input.value = "";
    button.style.opacity = 1;
    h4.style.opacity = 1;
  });

  const loader = document.querySelector("#loader");
  setTimeout(function () {
    loader.style.top = "-100%";
  }, 3300);

  document.getElementById("uploadForm")?.addEventListener("submit", function () {
    console.log("Form submitted!");
  });
});

function loco() {
  gsap.registerPlugin(ScrollTrigger);
  const locoScroll = new LocomotiveScroll({
    el: document.querySelector("#outer"),
    smooth: true,
    tablet: { smooth: true },
    smartphone: { smooth: true }
  });

  locoScroll.on("scroll", ScrollTrigger.update);

  ScrollTrigger.scrollerProxy("#outer", {
    scrollTop(value) {
      return arguments.length
        ? locoScroll.scrollTo(value, 0, 0)
        : locoScroll.scroll.instance.scroll.y;
    },
    getBoundingClientRect() {
      return {
        top: 0,
        left: 0,
        width: window.innerWidth,
        height: window.innerHeight
      };
    },
    pinType: document.querySelector("#outer").style.transform ? "transform" : "fixed"
  });

  ScrollTrigger.addEventListener("refresh", () => locoScroll.update());
  ScrollTrigger.refresh();
}