import "./App.css";
import React from "react";
import axios from "axios";
import { useFormik } from "formik";
import { Link } from "react-router-dom";

function App() {
  const handleSubmit = (values) => {
    console.log(values);
  };

  const formik = useFormik({
    initialValues: {
      name: "",
      message: "",
    },
    onSubmit: (values, { resetForm }) => {
      axios
        .post("http://127.0.0.1:5000/submit", values)
        .then((response) => {
          console.log(response);
          alert("Email sent successfully!");
          resetForm();
        })
        .catch((error) => {
          console.log(error);
          alert("Failed to send email. Please try again.");
        });
    },
  });

  return (
    <>
      <form onSubmit={formik.handleSubmit}>
        <input
          type="text"
          name="name"
          placeholder="Name"
          onChange={formik.handleChange}
          value={formik.values.name}
        />
        <input
          type="text"
          name="message"
          placeholder="Message"
          onChange={formik.handleChange}
          value={formik.values.message}
        />
        <button type="submit">Submit</button>
      </form>

      <Link to="/chat-bot">
        <button>Chat with Bot</button>
      </Link>
    </>
  );
}

export default App;
