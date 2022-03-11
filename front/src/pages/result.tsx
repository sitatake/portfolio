import { Center, Square, Circle, Button, VStack } from "@chakra-ui/react";
import React, { useState, useEffect, useContext } from "react";
import axios from "axios";
import { NameContext } from "../state/nameContext";
import { useRouter } from "next/router";

export default function Home() {
  const [result, setResult] = useState("");
  const { name, setName } = useContext(NameContext);
  const router = useRouter();
  const back = () => {
    router.push("/");
  };

  useEffect(() => {
    axios
      .get(`http://127.0.0.1:5000/uploads/${name}`)
      .catch()
      .then(({ data }) => setResult(data));
  }, []);

  return (
    <VStack spacing={10}>
      <Center bg="tomato" h="100px" w="full" color="white" fontSize="30">
        Portfolio
      </Center>
      <Center p="10">この画像が加工されている可能性は{result}％です！</Center>
      <Button colorScheme="orange" variant="outline" onClick={back}>
        戻る
      </Button>
    </VStack>
  );
}
