import React from "react";
import ReactDOM from "react-dom/client";
import { App } from "./App";
import "./index.css";

const app = <App />;
const root = ReactDOM.createRoot(document.getElementById("root")!);

if (import.meta.env.DEV) {
  root.render(app);
} else {
  root.render(
    <React.StrictMode>
      {app}
    </React.StrictMode>
  );
}
