import { Center, Square, Circle } from "@chakra-ui/react";
import React, { useState, useEffect, useContext } from "react";
import axios from "axios";
import { NameContext } from "../state/nameContext";

export default function Home() {
  const [result, setResult] = useState("");
  const { name, setName } = useContext(NameContext);

  useEffect(() => {
    axios
      .get(`http://127.0.0.1:5000/uploads/${name}`)
      .catch()
      .then(({ data }) => setResult(data));
  }, []);

  return (
    <>
      <Center bg="tomato" h="100px" color="white" fontSize="30">
        Portfolio
      </Center>
      <Center p="10">画像の名前は{result}です！</Center>
    </>
  );
}
