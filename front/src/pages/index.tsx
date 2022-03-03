import { Center, Container, Button, Input, VStack } from "@chakra-ui/react";
import { useRouter } from "next/router";
import React, { useState, useContext } from "react";
import axios from "axios";
import { NameContext } from "../state/nameContext";

export default function Home() {
  const router = useRouter();
  const [file, setFile] = useState<File>();
  const { name, setName } = useContext(NameContext);

  const ok = async (e: any) => {
    e.preventDefault();
    setName(file?.name);
    const body = new FormData();
    body.append("file", file);
    await axios.post("http://127.0.0.1:5000/upload", body).catch().then();
    router.push("/result");
  };

  return (
    <VStack spacing={10}>
      <Center bg="tomato" h="100px" w="full" color="white" fontSize="30">
        Portfolio
      </Center>
      <Container>
        当サイトはJPEG画像が加工されているかどうかをAIによって調べさせるサイトです．画像を入力し，加工されているか否かを判別させてみましょう．
      </Container>
      <form onSubmit={ok}>
        <Input
          type="file"
          onChange={(e: any) => setFile(e.target.files[0])}
        ></Input>
        <Button colorScheme="orange" m="4" type="submit" variant="outline">
          決定
        </Button>
      </form>
    </VStack>
  );
}
