document.addEventListener("DOMContentLoaded", function () {
  // Your existing image upload handling and loader code here...
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

  // Ensure removeBtn is hidden by default
  removeBtn.style.opacity = 0;
  removeBtn.style.pointerEvents = "none";

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
      removeBtn.style.pointerEvents = "auto";
      const submitBtn = document.getElementById("submitBtn");
      if (submitBtn) {
        submitBtn.style.opacity = 1;
        submitBtn.style.pointerEvents = "auto"; // Ensure it is clickable
      }
    } else {
      alert("Please select a valid image file.");
    }
  });

  removeBtn.addEventListener("click", function (event) {
    event.preventDefault();
    pre.src = "";
    pre.style.transition = "opacity 0.9s ease";
    pre.style.opacity = 0;
    box.style.display = "none";
    removeBtn.style.opacity = 0;

    const submitBtn = document.getElementById("submitBtn");
    if (submitBtn) {
      submitBtn.style.opacity = 0;
      submitBtn.style.pointerEvents = "none";
    }
    input.value = "";
    button.style.opacity = 1;
    h4.style.opacity = 1;

    setTimeout(() => {
      pre.style.display = "none";
      pre.style.opacity = 1;
      removeBtn.style.display = "none";
      removeBtn.style.opacity = 0;
      removeBtn.style.pointerEvents = "none";
    }, 900);
  });

  const loader = document.querySelector("#loader");
  setTimeout(function () {
    loader.style.top = "-100%";
  }, 4500);

  document.getElementById("uploadForm")?.addEventListener("submit", function (e) {
  e.preventDefault(); // 🔥 Ye line reload rokti hai!

  document.getElementById("result").innerText = "Loading";
  let dots = 0;
  const loadingInterval = setInterval(() => {
    const result = document.getElementById("result");
    dots = (dots + 1) % 4;
    result.innerText = "Loading" + ".".repeat(dots);
  }, 500);

  const formData = new FormData(this);

  fetch("/predict", {
    method: "POST",
    body: formData
  })
  .then(response => response.json())
  .then(data => {
    clearInterval(loadingInterval);
    const resultDiv = document.getElementById("result");
    resultDiv.style.display = "block";
    resultDiv.innerHTML = `
      <h2>Extracted Text:</h2>
      <pre style="max-width: 60%;height:100px">${data.text}</pre>
      <div>
        <h3>Download Options:</h3>
        <ul>
          <li><a href="${data.docx_link}" download class="download-link">Word</a></li>
          <li><a href="${data.pdf_link}" download class="download-link">PDF</a></li>
          <li><a href="${data.txt_link}" download class="download-link">Plain Text</a></li>
        </ul>
      </div>
    `;
  })
  .catch(error => {
    clearInterval(loadingInterval);
    console.error("Error during fetch:", error);
  });
});
});

function loco(){
    gsap.registerPlugin(ScrollTrigger);
  const locoScroll = new LocomotiveScroll({
    el: document.querySelector("#outer"),
    smooth: true,
    tablet: { smooth: true },
    smartphone: { smooth: true },
    scrollbar: {
    show: true // 🔧 Enable Locomotive's custom scrollbar
  }
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

  // Debug scroll limits
  console.log('Scroll limit:', locoScroll.scroll.instance.limit);

  // Update locoScroll on window resize to prevent scroll stalling
  window.addEventListener('resize', () => {
    locoScroll.update();
  });

}
loco();

document.addEventListener("DOMContentLoaded", () => {
  const preview = document.getElementById("preview");
  const imgOverlay = document.getElementById("imgOverlay");
  const overlayImg = document.getElementById("overlayImg");

  preview.addEventListener("click", () => {
    if (!preview.src) return; // no image loaded, do nothing

    overlayImg.src = preview.src; // set overlay image source
    imgOverlay.style.display = "flex"; // show overlay with flex to center image
    document.body.style.overflow = "hidden"; // prevent background scroll
  });

  imgOverlay.addEventListener("click", (e) => {
    if (e.target === imgOverlay) { // clicked on the overlay background (outside image)
      imgOverlay.style.display = "none";
      overlayImg.src = "";
      document.body.style.overflow = ""; // re-enable scroll
    }
  });
});
// Remove the separate loco() function and the bottom separate LocomotiveScroll instance and scroll listener to avoid duplicates