const form = document.querySelector("#ticket-form");
const ticket = document.querySelector("#ticket");
const submit = document.querySelector("#submit");
const status = document.querySelector("#status");
const result = document.querySelector("#result");

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = ticket.value.trim();
  if (!text) {
    status.textContent = "Enter a support ticket first.";
    return;
  }

  submit.disabled = true;
  status.textContent = "Extracting fields with the local model...";
  result.textContent = "";
  try {
    const response = await fetch("/api/extract", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || `Request failed (${response.status}).`);
    }
    result.textContent = JSON.stringify(payload, null, 2);
    status.textContent = "Extraction complete.";
  } catch (error) {
    status.textContent = `Could not extract ticket: ${error.message}`;
  } finally {
    submit.disabled = false;
  }
});
