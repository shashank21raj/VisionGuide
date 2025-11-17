import React, { useEffect } from "react";
import './CanvasBackground.css'

const CanvasBackground = () => {
  useEffect(() => {
    const canvas = document.getElementById("background-canvas");
    const ctx = canvas.getContext("2d");

    const resizeCanvas = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };

    resizeCanvas();

    let dots = [];

    // Create dots randomly across the screen
    for (let i = 0; i < 50; i++) {
      dots.push({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        radius: Math.random() * 3 + 1,
        dx: Math.random() * 2 - 1,
        dy: Math.random() * 2 - 1,
      });
    }

    let animationId;

    function animate() {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      for (let i = 0; i < dots.length; i++) {
        let dot = dots[i];

        dot.x += dot.dx * 0.7;
        dot.y += dot.dy * 0.7;

        if (dot.x < 0 || dot.x > canvas.width) dot.dx *= -1;
        if (dot.y < 0 || dot.y > canvas.height) dot.dy *= -1;

        ctx.beginPath();
        ctx.arc(dot.x, dot.y, dot.radius, 0, Math.PI * 2);
        ctx.fillStyle = "#bbb";
        ctx.fill();
      }

      // Connect dots with lines when close
      for (let i = 0; i < dots.length; i++) {
        for (let j = i + 1; j < dots.length; j++) {
          let dx = dots[i].x - dots[j].x;
          let dy = dots[i].y - dots[j].y;
          let distance = Math.sqrt(dx * dx + dy * dy);

          if (distance < 100) {
            ctx.beginPath();
            ctx.moveTo(dots[i].x, dots[i].y);
            ctx.lineTo(dots[j].x, dots[j].y);
            ctx.strokeStyle = "rgba(180, 180, 180, 0.2)";
            ctx.lineWidth = 0.9;
            ctx.stroke();
          }
        }
      }

      animationId = requestAnimationFrame(animate);
    }

    animate();

    const handleResize = () => {
      resizeCanvas();
    };

    window.addEventListener('resize', handleResize);

    return () => {
      cancelAnimationFrame(animationId);
      window.removeEventListener('resize', handleResize);
    };
  }, []);

  return <canvas id="background-canvas" className="background-canvas"></canvas>;
};

export default CanvasBackground;