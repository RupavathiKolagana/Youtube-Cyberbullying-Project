function showSection(id) {
    document.querySelectorAll('.section').forEach(section => {
        section.classList.remove('active');
    });

    const activeSection = document.getElementById(id);
    activeSection.classList.add('active');
    activeSection.scrollTop = 0;
}

function loadData() {
    fetch("/abusive-comments")
        .then(res => res.json())
        .then(data => {

            /* ---------- STATUS ---------- */
            document.getElementById("status").textContent =
                data.status || "Unknown";


            /* ---------- COUNTERS ---------- */
            document.getElementById("processed-videos").textContent =
                `${data.processed_videos || 0} / ${data.total_videos || 0}`;

            document.getElementById("total-comments").textContent =
                data.total_comments || 0;


            /* ---------- ABUSIVE COUNT ---------- */
            let abusiveCount = 0;

            if (data.abusive_comments !== undefined) {
                abusiveCount = data.abusive_comments;
            } else if (data.all_comments) {
                abusiveCount = data.all_comments.filter(c => (c.score || 0) >= 0.6).length;
            }

            document.getElementById("abusive-count").textContent = abusiveCount;


            /* ---------- TABLE ---------- */
            let rows = "";

            if (data.all_comments && data.all_comments.length > 0) {

                data.all_comments.forEach(c => {

                    let score = (c.score || 0);

                    /* Confidence Color Logic */
                    let color = "#10b981"; // green
                    if (score >= 0.8) color = "#ef4444"; // red
                    else if (score >= 0.6) color = "#f59e0b"; // orange

                    /* Row highlight if abusive */
                    let rowStyle = score >= 0.6
                        ? "style='background:#fff1f2;'"
                        : "";

                    rows += `
                        <tr ${rowStyle}>
                            <td>${c.video_id || "-"}</td>
                            <td>${c.comment_text || ""}</td>
                            <td>${c.author || "Unknown"}</td>
                            <td style="font-weight:bold;color:${color}">
                                ${score.toFixed(4)}
                            </td>
                            <td>
                                ${c.profile_url && c.profile_url !== "#"
                                    ? `<a href="${c.profile_url}" target="_blank">View</a>`
                                    : `Not Available`
                                }
                            </td>
                        </tr>
                    `;
                });

            } else {
                rows = `
                    <tr>
                        <td colspan="5" style="text-align:center;">
                            No comments available
                        </tr>
                `;
            }

            document.getElementById("results-table").innerHTML = rows;

        })
        .catch(err => {
            console.error("Error loading data:", err);
            document.getElementById("status").textContent = "Server Error";
        });
}

/* Auto refresh every 3 sec */
setInterval(loadData, 3000);

/* Initial load */
loadData();